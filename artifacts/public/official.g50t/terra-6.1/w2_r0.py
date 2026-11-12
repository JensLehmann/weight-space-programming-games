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

    W1, b1 = p["W1"], p["b1"]
    W2, b2 = p["W2"], p["b2"]
    W3, b3 = p["W3"], p["b3"]
    Wh, Uh = p["Wh"], p["Uh"]

    # Order: up, down, left, right.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # W1 channels 0..3: detect the player two pixels in each
    # direction from the output location.
    for i, (dy, dx) in enumerate(directions):
        cy, cx = -2 * dy, -2 * dx
        W1[i, 5, 2 + cy, 2 + cx] = 1.0
        if dy:
            for side in (-1, 0, 1):
                W1[i, 9, 2 + cy + dy, 2 + side] = 1.0
            for side in (-1, 1):
                W1[i, 9, 2 + cy, 2 + side] = 1.0
        else:
            for side in (-1, 0, 1):
                W1[i, 9, 2 + side, 2 + cx + dx] = 1.0
            for side in (-1, 1):
                W1[i, 9, 2 + side, 2 + cx] = 1.0
        b1[i] = -5.0

    # Visible non-background terrain. This is only an approximation
    # to the game's actual walkability and collision rules.
    for color in range(1, 16):
        W1[4, color, 2, 2] = 1.0

    # Copies of color 9 displaced by two pixels, and color 5.
    for channel, (dy, dx) in enumerate(directions, start=5):
        W1[channel, 9, 2 + 2 * dy, 2 + 2 * dx] = 1.0
    W1[9, 5, 2, 2] = 1.0

    # Player center: eight 9s around a 5.
    W1[10, 9, 1:4, 1:4] = 1.0
    W1[10, 9, 2, 2] = 0.0
    W1[10, 5, 2, 2] = 1.0
    b1[10] = -8.0

    # Switch-like 3x3 patches with four surrounding color-5 corners.
    for channel, color in zip((11, 12, 13), (8, 11, 15)):
        W1[channel, color, 1:4, 1:4] = 1.0
        for dy in (-2, 2):
            for dx in (-2, 2):
                W1[channel, 5, 2 + dy, 2 + dx] = 1.0
        b1[channel] = -12.0

    # Wire-colored pixels two pixels from the output location.
    for i, (dy, dx) in enumerate(directions):
        for color in (8, 11, 15):
            W1[14 + i, color, 2 + 2 * dy, 2 + 2 * dx] = 1.0

    # Player sampling markers at offsets of three pixels.
    for i, (dy, dx) in enumerate(directions):
        W2[i, i, 1 - dy, 1 - dx] = 1.0
    W2[4, 4, 1, 1] = 1.0

    # Goal-like center: color 5 at the center and its four neighbors,
    # with color 9 three pixels away on opposite sides.
    for out, first, second, offsets in (
        (5, 5, 6, ((0, 1), (2, 1))),
        (6, 7, 8, ((1, 0), (1, 2))),
    ):
        W2[out, first, offsets[0][0], offsets[0][1]] = 1.0
        W2[out, second, offsets[1][0], offsets[1][1]] = 1.0
        for ky, kx in ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2)):
            W2[out, 9, ky, kx] = 1.0
        b2[out] = -6.0

    for channel in (11, 12, 13):
        W2[7, channel, 1, 1] = 1.0

    # Approximate "standing on a wired switch" test.
    for i, (dy, dx) in enumerate(directions):
        W2[8 + i, 10, 1, 1] = 1.0
        W2[8 + i, 14 + i, 1 + dy, 1 + dx] = 1.0
        b2[8 + i] = -1.0

    # W3 channel 0 is a background mask. Channels 1..4 are
    # complements of the directional player sampling markers.
    b3[0] = 1.0
    W3[0, 4] = -1.0
    for i in range(4):
        b3[1 + i] = 1.0
        W3[1 + i, i] = -1.0

    W3[5, 5] = 1.0
    W3[5, 6] = 1.0
    W3[6, 7] = 1.0
    for i in range(4):
        W3[7, 8 + i] = 1.0

    # h[0] diffuses the goal and switch signals. Background is clamped
    # by a large negative drive. h[1:5] sample the field near the player.
    Wh[0, 0, 1, 1] = 0.20
    for ky, kx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        Wh[0, 0, ky, kx] = 0.1975
    Uh[0, 0] = -1_000_000.0
    Uh[0, 5] = 1.0
    Uh[0, 6] = 2.0

    for i in range(4):
        Wh[1 + i, 0, 1, 1] = 1.0
        Uh[1 + i, 1 + i] = -1_000_000.0

    # g begins with the 16 spatial maxima of h, followed by its means,
    # then the maxima and means of f3.
    for i in range(4):
        p["Wz"][i, 1 + i] = 1.0
    p["Wz"][4, 32 + 7] = 1.0

    # Remember whether ACTION5 has already been attempted.
    p["Wm"][0, 0] = 5.0
    p["Wma"][0, 5] = 5.0

    # Directional logits are scaled down so the switch test can win.
    for i, action in enumerate((1, 2, 3, 4)):
        p["Wo"][action, i] = 0.0001
    p["Wo"][5, 4] = 8.0
    p["Wo"][5, 128] = -8.0
    p["bo"][5] = -4.0

    # Avoid reset and unused actions if they are available.
    p["bo"][0] = -10.0
    p["bo"][6] = -10.0
    p["bo"][7] = -10.0
    return p
