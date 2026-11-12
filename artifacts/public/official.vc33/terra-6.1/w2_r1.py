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
    W2 = p["W2"]
    W3 = p["W3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    # Player edge, classified by the side containing its colour-4 body.
    body_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for c, (dy, dx) in enumerate(body_offsets):
        W1[c, 11, 2, 2] = 1
        W1[c, 14, 2, 2] = 1
        W1[c, 4, 2 + dy, 2 + dx] = 1
        b1[c] = -1

    # Marker embedded in a vertical or horizontal wall.
    W1[4, 11, 2, 2] = 1
    W1[4, 14, 2, 2] = 1
    W1[4, 5, 1, 2] = 1
    b1[4] = -1

    W1[5, 11, 2, 2] = 1
    W1[5, 14, 2, 2] = 1
    W1[5, 5, 2, 1] = 1
    b1[5] = -1

    W1[6, 9, 2, 2] = 1

    # Controls adjacent to wall above, below, left, or right.
    for k, (dy, dx) in enumerate(
        [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ):
        c = 7 + k
        W1[c, 9, 2, 2] = 1
        W1[c, 5, 2 + dy, 2 + dx] = 1
        b1[c] = -1

    # A control touching colour 0 along each possible gravity axis.
    W1[11, 9, 2, 2] = 1
    W1[11, 0, 2, 1] = 1
    W1[11, 0, 2, 3] = 1
    b1[11] = -1

    W1[12, 9, 2, 2] = 1
    W1[12, 0, 1, 2] = 1
    W1[12, 0, 3, 2] = 1
    b1[12] = -1

    # One edge pixel per player.
    for c in range(4):
        W2[c, c, 1, 1] = 1
        if c < 2:
            W2[c, c, 1, 0] = -1
        else:
            W2[c, c, 0, 1] = -1

    # One corner per wall marker.
    W2[4, 4, 1, 1] = 1
    W2[4, 4, 1, 0] = -1
    W2[5, 5, 1, 1] = 1
    W2[5, 5, 0, 1] = -1

    for c in range(6, 13):
        W2[c, c, 1, 1] = 1

    # Axial player projections.
    for c in range(4):
        W3[0, 32 + c] = 1
        W3[1, 64 + c] = 1

    # Axial marker projections.
    for c in (4, 5):
        W3[2, 32 + c] = 1
        W3[3, 64 + c] = 1

    W3[4, 6] = 1
    for k in range(4):
        W3[5 + k, 7 + k] = 1

    # Broadcast platform availability across the control's row or column.
    W3[9, 32 + 11] = 1
    W3[10, 64 + 12] = 1

    # Directional player and marker waves.
    decay = 0.985
    for source, base in ((0, 0), (2, 4)):
        Uh[base, source] = 1
        Wh[base, base, 0, 1] = decay

        Uh[base + 1, source] = 1
        Wh[base + 1, base + 1, 2, 1] = decay

    for source, base in ((1, 2), (3, 6)):
        Uh[base, source] = 1
        Wh[base, base, 1, 0] = decay

        Uh[base + 1, source] = 1
        Wh[base + 1, base + 1, 1, 2] = decay

    # Control plane, player-side wave, gravity-axis comparison waves,
    # and whether the control is on the player's own side.
    candidates = [
        (6, 1, (2, 3, 6, 7), False),
        (6, 0, (2, 3, 6, 7), True),
        (5, 1, (2, 3, 6, 7), True),
        (5, 0, (2, 3, 6, 7), False),
        (8, 3, (0, 1, 4, 5), False),
        (8, 2, (0, 1, 4, 5), True),
        (7, 3, (0, 1, 4, 5), True),
        (7, 2, (0, 1, 4, 5), False),
    ]

    for out, (plane, cross_wave, waves, own_side) in enumerate(
        candidates, start=8
    ):
        player_a, player_b, marker_a, marker_b = waves
        direction = 1.0 if own_side else -1.0

        Uh[out, plane] = 20
        Wh[out, cross_wave, 1, 1] += 2

        Wh[out, player_a, 1, 1] += 8 * direction
        Wh[out, player_b, 1, 1] += 8 * direction
        Wh[out, marker_a, 1, 1] -= 8 * direction
        Wh[out, marker_b, 1, 1] -= 8 * direction

        floor_channel = 9 if out < 12 else 10
        Uh[out, floor_channel] = -2
        bh[out] = -18

    # Base score keeps clicks on controls.
    p["Wc"][0, 4] = 100
    for c in range(8, 16):
        p["Wc"][0, 32 + c] = 1

    p["bo"][6] = 1
    return p
