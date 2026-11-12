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

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    floor_colors = (2, 5, 8, 9, 11, 15)
    switch_colors = (8, 11, 15)

    # Erode the visible floor by two pixels in each cardinal direction.
    # This helps reject narrow decorative wires outside the board.
    for dy, dx in [(0, 0), (-2, 0), (2, 0), (0, -2), (0, 2)]:
        for color in floor_colors:
            W1[0, color, 2 + dy, 2 + dx] = 1
    b1[0] = -4

    # Goal center.
    W1[1, 9, 2, 2] = 1
    for dy, dx in directions:
        W1[1, 5, 2 + dy, 2 + dx] = 1
    b1[1] = -4

    # Player markers displaced two pixels in each direction.
    for i, (dy, dx) in enumerate(directions):
        ch = 2 + i
        py, px = -dx, dy
        cy, cx = -2 * dy, -2 * dx

        W1[ch, 5, 2 + cy, 2 + cx] = 1
        for distance in (1, 2):
            for side in (-1, 0, 1):
                yy = cy + distance * dy + side * py
                xx = cx + distance * dx + side * px
                W1[ch, 9, 2 + yy, 2 + xx] = 1
        for side in (-1, 1):
            yy = cy + side * py
            xx = cx + side * px
            W1[ch, 9, 2 + yy, 2 + xx] = 1
        b1[ch] = -8

    # Dense colored patches distinguish gates from 3x3 switches.
    for ch, color in ((6, 8), (7, 11)):
        W1[ch, color, :, :] = 1
        b1[ch] = -12

    # An unoccupied switch contains a solid 3x3 colored patch.
    for color in switch_colors:
        W1[8, color, 1:4, 1:4] = 1
    b1[8] = -8

    for i, (dy, dx) in enumerate(directions):
        py, px = -dx, dy

        # Sample two rows of three pixels across the destination.
        # W2 shifts this detector four pixels from the player.
        ch = 9 + i
        for distance in (0, 2):
            for side in (-2, 0, 2):
                yy = distance * dy + side * py
                xx = distance * dx + side * px
                for color in floor_colors:
                    W1[ch, color, 2 + yy, 2 + xx] = 1
        b1[ch] = -5

        # Partial switch at the destination.
        ch = 13 + i
        for distance in (1, 2):
            for side in (-1, 0, 1):
                yy = distance * dy + side * py
                xx = distance * dx + side * px
                for color in switch_colors:
                    W1[ch, color, 2 + yy, 2 + xx] = 1
        b1[ch] = -5

        # Near edge of a closed gate.
        for base, color in ((20, 8), (24, 11)):
            ch = base + i
            for side in (-2, 0, 2):
                yy = side * py
                xx = side * px
                W1[ch, color, 2 + yy, 2 + xx] = 1
            b1[ch] = -2

    # Current recording-slot icon and unused slot to its right.
    W1[17, 0, 2, 2] = 1
    for dy, dx in directions:
        W1[17, 9, 2 + dy, 2 + dx] = 1
    b1[17] = -4
    W1[18, 1, 2, 4] = 1

    # Traversable map, with a dilated gate mask.
    W2[0, 0, 1, 1] = 1
    W2[0, 6, :, :] = -1
    W2[0, 7, :, :] = -1

    W2[1, 1, 1, 1] = 1

    # Switch targets, excluding gate interiors and edges.
    W2[2, 8, 1, 1] = 1
    W2[2, 6, :, :] = -1
    W2[2, 7, :, :] = -1

    for i, (dy, dx) in enumerate(directions):
        # Plausible move marker, three pixels from the player.
        ch = 3 + i
        W2[ch, 2 + i, 1 - dy, 1 - dx] = 1
        W2[ch, 9 + i, 1 + dy, 1 + dx] = 1
        W2[ch, 20 + i, 1 + dy, 1 + dx] = -1
        W2[ch, 24 + i, 1 + dy, 1 + dx] = -1
        b2[ch] = -1

        # Marker for entering a switch.
        ch = 7 + i
        W2[ch, 2 + i, 1 - dy, 1 - dx] = 1
        W2[ch, 13 + i, 1 + dy, 1 + dx] = 1
        W2[ch, 20 + i, 1 + dy, 1 + dx] = -1
        W2[ch, 24 + i, 1 + dy, 1 + dx] = -1
        b2[ch] = -1

    W2[11, 17, 1, 1] = 1
    W2[11, 18, 1, 2] = 1
    b2[11] = -1

    for i in range(12):
        p["W3"][i, i] = 1

    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]
    offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

    # Two clipped reachability floods, one from the goal and one
    # from visible switches. Count channels encode arrival time.
    for a, b, count, source in ((0, 1, 2, 1), (3, 4, 5, 2)):
        for out in (a, b):
            for dy, dx in offsets:
                Wh[out, a, 1 + dy, 1 + dx] = 1
                Wh[out, b, 1 + dy, 1 + dx] = -1
            Uh[out, source] = 1
            bh[out] = -1
        Uh[a, 0] = 1

        Wh[count, count, 1, 1] = 1
        Wh[count, a, 1, 1] = 1
        Wh[count, b, 1, 1] = -1

    # Read the two distance fields at the move markers.
    for i in range(4):
        for out, count in ((6 + i, 2), (10 + i, 5)):
            Wh[out, count, 1, 1] = 1
            Uh[out, 3 + i] = 128
            bh[out] = -128

    Wz = p["Wz"]

    for i in range(4):
        Wz[i, 32 + 7 + i] = 1
    Wz[4, 32 + 11] = 1

    for i in range(4):
        Wz[5 + i, 6 + i] = 1 / 128

        # Prefer a reachable goal over switches. If no neighboring
        # move has a goal signal, retain the switch-distance signal.
        Wz[9 + i, 10 + i] = 1 / 128
        for j in range(4):
            Wz[9 + i, 6 + j] = -2

        Wz[13 + i, 32 + 3 + i] = 1

    Wm, Wmz, Wma, bm = p["Wm"], p["Wmz"], p["Wma"], p["bm"]

    # Store switch-entry opportunities from the previous frame.
    for i in range(4):
        Wmz[i, i] = 5

    # Trigger rewind after entering a switch if a slot remains.
    for i in range(4):
        Wm[4 + i, i] = 4
        Wma[4 + i, 1 + i] = 4
        Wmz[4 + i, 4] = 4
        bm[4 + i] = -9.5

    for i in range(4):
        Wma[8 + i, 1 + i] = 3

    Wo, bo = p["Wo"], p["bo"]
    opposite = [1, 0, 3, 2]

    for i in range(4):
        action = 1 + i
        Wo[action, 5 + i] = 100
        Wo[action, 9 + i] = 200
        Wo[action, 13 + i] = 0.1
        Wo[action, 128 + 8 + i] = 0.02
        Wo[action, 128 + 8 + opposite[i]] = -0.04

    for i in range(4):
        Wo[5, 128 + 4 + i] = 100000
    bo[5] = 300000

    bo[0] = -1000000
    bo[6] = -1000000
    bo[7] = -1000000

    return p
