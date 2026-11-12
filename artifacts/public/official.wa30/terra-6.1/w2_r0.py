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
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Player templates, anchored at tile top-left plus (1, 1).
    for d in range(4):
        for y in range(4):
            for x in range(4):
                edge = (
                    (d == 0 and y == 0)
                    or (d == 1 and y == 3)
                    or (d == 2 and x == 0)
                    or (d == 3 and x == 3)
                )
                W1[d, 0 if edge else 14, y + 1, x + 1] = 2.0
        b1[d] = -31.0

    # Crate and player-carried crate templates.
    for y in range(4):
        for x in range(4):
            core = 1 <= y <= 2 and 1 <= x <= 2
            if core:
                W1[4, 9, y + 1, x + 1] = 2.0
                W1[5, 9, y + 1, x + 1] = 2.0
            else:
                for color in (0, 3, 4, 5):
                    W1[4, color, y + 1, x + 1] = 2.0
                W1[5, 0, y + 1, x + 1] = 2.0
    b1[4] = -31.0
    b1[5] = -31.0

    # Solid wall interior.
    W1[6, 5, 1:4, 1:4] = 2.0
    b1[6] = -17.0

    edge_points = [
        [(-2, 0), (-2, 1)],
        [(2, 0), (2, 1)],
        [(0, -2), (1, -2)],
        [(0, 2), (1, 2)],
    ]
    shifts = [(0, 0), (1, 0), (0, 0), (0, 1)]

    for d, points in enumerate(edge_points):
        # Adjacent visible goal tile.
        for dy, dx in points:
            for color in (2, 9):
                W1[7 + d, color, dy + 2, dx + 2] = 2.0
        b1[7 + d] = -3.0

        # Adjacent solid actor, wall, or outside the board.
        for dy, dx in points:
            for color in (5, 11, 12, 15):
                W1[11 + d, color, dy + 2, dx + 2] = 2.0
            W1[11 + d, 18, dy + 2, dx + 2] = -2.0
        b1[11 + d] = 1.0

        # Ordinary crate edge.
        for dy, dx in points:
            for color in (3, 4):
                W1[15 + d, color, dy + 2, dx + 2] = 2.0
        b1[15 + d] = -3.0

        # Crate or removable adversary edge.
        for dy, dx in points:
            for color in (3, 4, 11, 15):
                W1[19 + d, color, dy + 2, dx + 2] = 2.0
        b1[19 + d] = -3.0

        # Patterned barrier edge.
        colors = (2, 1) if d in (0, 2) else (1, 2)
        for (dy, dx), color in zip(points, colors):
            W1[27 + d, color, dy + 2, dx + 2] = 2.0
        b1[27 + d] = -3.0

    # Visible goal interiors near a 9-colored boundary.
    for d, (dy, dx) in enumerate(directions):
        ch = 23 + d
        W1[ch, 2, 2, 2] = 2.0
        W1[ch, 9, 2 + dy, 2 + dx] = 1.0
        W1[ch, 9, 2 + 2 * dy, 2 + 2 * dx] = 1.0
        b1[ch] = -2.5

    for d in range(4):
        W2[0, d, 1, 1] = 1.0
    W2[1, 4, 1, 1] = 1.0
    W2[2, 5, 1, 1] = 1.0
    W2[3, 6, :, :] = 2.0
    b2[3] = -1.0

    for d, (sy, sx) in enumerate(shifts):
        for orientation in range(4):
            W2[4 + d, orientation, 1, 1] = 2.0
            W2[8 + d, orientation, 1, 1] = 2.0

        # Blocked player destination.
        for ch in (11 + d, 15 + d, 27 + d):
            W2[4 + d, ch, 1 + sy, 1 + sx] = 2.0
            W2[8 + d, ch, 1 + sy, 1 + sx] = 2.0
        b2[4 + d] = -3.0
        b2[8 + d] = -3.0

        # Blocked carried-crate destination.
        W2[12 + d, 5, 1, 1] = 2.0
        for ch in (11 + d, 15 + d, 27 + d):
            W2[12 + d, ch, 1 + sy, 1 + sx] = 2.0
        b2[12 + d] = -3.0

        # Visible goal adjacent to the carried crate.
        W2[16 + d, 5, 1, 1] = 2.0
        W2[16 + d, 7 + d, 1 + sy, 1 + sx] = 2.0
        b2[16 + d] = -3.0

        # Player faces an interactable object.
        W2[27 + d, d, 1, 1] = 2.0
        W2[27 + d, 19 + d, 1 + sy, 1 + sx] = 2.0
        b2[27 + d] = -3.0

    for d in range(4):
        W2[22 + d, 23 + d, 1, 1] = 1.0

    # A likely delivered crate must be an actual crate and have at least
    # one visible neighboring goal tile.
    W2[26, 4, 1, 1] = 100.0
    for d, (sy, sx) in enumerate(shifts):
        W2[26, 7 + d, 1 + sy, 1 + sx] = 2.0
    b2[26] = -101.0

    W3[0, 0] = 1.0                       # player
    W3[1, 1] = 1.0                       # target crate
    W3[1, 2] = -1.0
    W3[1, 26] = -1.0
    W3[2, 3] = 1.0                       # wall
    W3[4, 2] = 1.0                       # carried crate

    for d in range(4):
        W3[3, 27 + d] = 1.0              # interaction candidate
        W3[5, 22 + d] = 1.0              # goal source
        W3[6 + d, 4 + d] = 1.0           # blocked when free
        W3[10 + d, 8 + d] = 1.0          # blocked when carrying
        W3[10 + d, 12 + d] = 1.0
        W3[14 + d, 16 + d] = 1.0        # goal next to carried crate

    # Their difference caps the delivered-crate source at one.
    W3[18, 26] = 1.0
    W3[19, 26] = 1.0
    b3[19] = -1.0

    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    # Contractive fields for target crates and visible goals.
    for field, source, strength in ((0, 1, 0.4), (1, 5, 0.02)):
        for dy, dx in directions:
            Wh[field, field, 1 + dy, 1 + dx] = 0.249
        Uh[field, source] = strength
        Uh[field, 2] = -10.0

    for d, (dy, dx) in enumerate(directions):
        # Crate field sampled at the player.
        Wh[2 + d, 0, 1 + dy, 1 + dx] = 1.0
        Uh[2 + d, 0] = 10.0
        bh[2 + d] = -10.0

        # Goal field sampled at the carried crate.
        Wh[6 + d, 1, 1 + dy, 1 + dx] = 1.0
        Uh[6 + d, 4] = 10.0
        bh[6 + d] = -10.0

    # Rapidly decaying local field around likely delivered crates.
    for dy, dx in directions:
        Wh[10, 10, 1 + dy, 1 + dx] = 0.1
    Uh[10, 18] = 1000.0
    Uh[10, 19] = -1000.0
    Uh[10, 2] = -10000.0

    # Sample the local field when facing an interactable object.
    Wh[11, 10, 1, 1] = 1.0
    Uh[11, 3] = 10.0
    bh[11] = -10.06

    # g[0:16] are maxima of h; g[32:64] are maxima of f3.
    Wz, bz = p["Wz"], p["bz"]
    for d in range(4):
        # Seek a crate while free.
        Wz[d, 2 + d] = 1.0
        Wz[d, 32 + 4] = -4.0
        Wz[d, 32 + 6 + d] = -4.0

        # Seek a goal while carrying.
        Wz[4 + d, 6 + d] = 1.0
        Wz[4 + d, 32 + 4] = 4.0
        Wz[4 + d, 32 + 10 + d] = -4.0
        bz[4 + d] = -4.0

        # Goal-adjacent move that is not visibly blocked.
        Wz[9 + d, 32 + 14 + d] = 1.0
        Wz[9 + d, 32 + 10 + d] = -2.0

        # Explicit blocked-move penalties.
        Wz[14 + d, 32 + 6 + d] = 1.0
        Wz[14 + d, 32 + 4] = -2.0
        Wz[18 + d, 32 + 10 + d] = 1.0
        Wz[18 + d, 32 + 4] = 1.0
        bz[18 + d] = -1.0

    # Interaction candidate, suppressed near likely delivered crates.
    Wz[8, 32 + 3] = 1.0
    Wz[8, 32 + 4] = -1.0
    Wz[8, 11] = -20.0
    Wz[13, 32 + 4] = 1.0

    # Store goal-adjacency flags, then detect a move into such a tile.
    for d in range(4):
        p["Wmz"][d, 9 + d] = 4.0
        p["Wm"][4 + d, d] = 4.0
        p["Wma"][4 + d, 1 + d] = 4.0
        p["bm"][4 + d] = -6.0

    # Detect ACTION5 immediately after an observation with a carried crate.
    p["Wmz"][8, 13] = 4.0
    p["Wm"][9, 8] = 4.0
    p["Wma"][9, 5] = 4.0
    p["bm"][9] = -6.0

    # Remember the previous direction for a small reversal penalty.
    for d in range(4):
        p["Wma"][10 + d, 1 + d] = 2.0

    opposite = [1, 0, 3, 2]
    for d in range(4):
        p["Wo"][1 + d, d] = 1.0
        p["Wo"][1 + d, 4 + d] = 1.0
        p["Wo"][1 + d, 14 + d] = -2.0
        p["Wo"][1 + d, 18 + d] = -2.0
        p["Wo"][1 + d, 128 + 10 + opposite[d]] = -0.04

    # Grab, remove an adversary, or release after entering a goal.
    p["Wo"][5, 8] = 8.0
    for d in range(4):
        p["Wo"][5, 128 + 4 + d] = 4.0
    p["Wo"][5, 128 + 9] = -4.0
    p["bo"][5] = 11.0

    p["bo"][0] = -1000.0
    p["bo"][6] = -1000.0
    p["bo"][7] = -1000.0

    return p
