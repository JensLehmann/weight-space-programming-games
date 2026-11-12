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

    # Target mask, with rightmost UI column and bottom row removed.
    p["W1"][0, 11, 2, 2] = 1
    p["W1"][1, 11, 2, 2] = 1
    p["W1"][1, 17, 2, 2] = 63
    p["b1"][1] = -63
    p["W1"][2, 11, 2, 2] = 1
    p["W1"][2, 16, 2, 2] = 63
    p["b1"][2] = -63

    # Original and reflected piece mask.
    for c in (4, 5):
        p["W1"][3, c, 2, 2] = 1
        p["W1"][4, c, 2, 2] = 1
        p["W1"][5, c, 2, 2] = 1
    p["W1"][4, 17, 2, 2] = 63
    p["b1"][4] = -63
    p["W1"][5, 16, 2, 2] = 63
    p["b1"][5] = -63

    p["W2"][0, 0, 1, 1] = 1
    p["W2"][0, 1, 1, 1] = -1
    p["W2"][0, 2, 1, 1] = -1
    p["W2"][1, 3, 1, 1] = 1
    p["W2"][1, 4, 1, 1] = -1
    p["W2"][1, 5, 1, 1] = -1
    p["W3"][0, 0] = 1
    p["W3"][1, 1] = 1

    # Coordinate moments of original pieces and mirror lines.
    # ReLU(mask + coordinate - 1) gates the coordinate by the mask.
    for out, colour, coord in (
        (6, 5, 17), (7, 5, 16),
        (8, 10, 17), (9, 10, 16),
    ):
        p["W1"][out, colour, 2, 2] = 1
        p["W1"][out, coord, 2, 2] = 1
        p["b1"][out] = -1

    # UI-column contributions to the colour-5 moments.
    p["W1"][10, 5, 2, 2] = 1
    p["W1"][10, 17, 2, 2] = 63
    p["b1"][10] = -63

    p["W1"][11, 5, 2, 2] = 1
    p["W1"][11, 17, 2, 2] = 63
    p["W1"][11, 16, 2, 2] = 1
    p["b1"][11] = -64

    for j in range(4):
        p["W2"][2 + j, 6 + j, 1, 1] = 1
        p["W3"][2 + j, 2 + j] = 1
    p["W2"][2, 10, 1, 1] = -1
    p["W2"][3, 11, 1, 1] = -1

    # Discounted directional target field.
    q = 0.97
    p["Wh"][0, 0, 2, 1] = q
    p["Wh"][1, 1, 0, 1] = q
    p["Uh"][0, 0] = 0.01
    p["Uh"][1, 0] = 0.01

    p["Wh"][2, 2, 1, 2] = q
    p["Wh"][3, 3, 1, 0] = q
    for out in (2, 3):
        p["Wh"][out, 0, 1, 1] = 0.5
        p["Wh"][out, 1, 1, 1] = 0.5

    # Sample the field on visible pieces and reflections.
    p["Wh"][4, 2, 1, 1] = 1
    p["Wh"][4, 3, 1, 1] = 1
    p["Uh"][4, 1] = 100
    p["bh"][4] = -100

    # Attraction score and uncovered-target penalty.
    p["Wz"][0, 16 + 4] = 500
    p["Wz"][1, 64 + 0] = 1500

    # Small encoding of the previous score.
    p["Wmz"][0, 0] = 1e-5
    p["Wmz"][0, 1] = -1e-5

    for action in range(1, 5):
        # Progress following that action.
        p["Wm"][action, 0] = -2_000_000
        p["Wmz"][action, 0] = 20
        p["Wmz"][action, 1] = -20
        p["Wma"][action, action] = 10000
        p["bm"][action] = -10000.003

        # Previous-action indicator.
        p["Wma"][4 + action, action] = 10

        # Remember the attempted direction across an undo.
        d = 8 + action
        p["Wm"][d, d] = 10
        p["Wma"][d, action] = 30
        p["Wma"][d, 7] = 10
        p["bm"][d] = -15

        u = 12 + action
        p["Wm"][u, d] = 10
        p["Wma"][u, 7] = 10
        p["bm"][u] = -15

    # Store four spatial moments in an approximately linear range.
    for j in range(4):
        p["Wz"][2 + j, 64 + 2 + j] = 1
        p["Wmz"][20 + j, 2 + j] = 0.004096

    # For each attempted direction, compare current and previous
    # moments. Both signs are tested to recognize unchanged frames.
    for action in range(1, 5):
        for j in range(4):
            for sign_index, sign in enumerate((1, -1)):
                unit = 24 + (action - 1) * 8 + j * 2 + sign_index
                p["Wm"][unit, 20 + j] = -sign * 100_000_000
                p["Wmz"][unit, 2 + j] = sign * 409600
                p["Wma"][unit, action] = 100000
                p["bm"][unit] = -99999.8

    def add_progress(logit, action, coefficient):
        p["Wo"][logit, 128 + action] += coefficient
        p["Wo"][logit, 128 + 4 + action] -= coefficient
        p["bo"][logit] += coefficient

    # Continue improving moves; undo score-decreasing moves.
    for action in range(1, 5):
        p["Wo"][action, 128 + 4 + action] += 2
        add_progress(action, action, 1)

        p["Wo"][7, 128 + 4 + action] += 2
        add_progress(7, action, -1)

    # Following an undo, try the next direction or change selection.
    for action in range(1, 5):
        destination = action + 1 if action < 4 else 5
        p["Wo"][destination, 128 + 12 + action] += 1.5
        p["bo"][destination] += 1.5

    # If the attempted move left the spatial moments unchanged,
    # skip directly to the next direction instead of undoing.
    for action in range(1, 5):
        destination = action + 1 if action < 4 else 5
        for j in range(4):
            for sign_index in range(2):
                unit = 24 + (action - 1) * 8 + j * 2 + sign_index
                p["Wo"][destination, 128 + unit] += 20
                p["bo"][destination] += 20
        p["Wo"][destination, 128 + 4 + action] -= 20 * 9.35

    p["bo"][1] += 0.005
    p["bo"][0] = -100
    p["bo"][6] = -100
    return p
