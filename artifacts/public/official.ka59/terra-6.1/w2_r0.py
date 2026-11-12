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

    # Selected piece marker, including black contact edges.
    w = p["W1"][0]
    w[0, 2, 2] = 1
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        w[0, 2 + dy, 2 + dx] = 1
        w[14, 2 + dy, 2 + dx] = 1
    p["b1"][0] = -4

    # Empty 5x5 goal. Its outline may be partly covered by a piece.
    w = p["W1"][1]
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if abs(dy) == 2 or abs(dx) == 2:
                for color in (0, 4, 5, 14):
                    w[color, 2 + dy, 2 + dx] = 1
    w[1, 2, 2] = 1
    p["b1"][1] = -16

    # Completed 5x5 goal, with selected or unselected centre.
    for ch, centers in ((2, (0,)), (3, (4, 5))):
        w = p["W1"][ch]
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if abs(dy) == 2 or abs(dx) == 2:
                    w[4, 2 + dy, 2 + dx] = 1
        for color in centers:
            w[color, 2, 2] = 1
        p["b1"][ch] = -16

    # Upper-left centre pixel of an unselected piece.
    w = p["W1"][4]
    w[4, 2, 2] = 1
    w[5, 2, 2] = 1
    for dy, dx in ((-1, 0), (0, -1)):
        w[0, 2 + dy, 2 + dx] = 1
        w[14, 2 + dy, 2 + dx] = 1
    for dy, dx in ((1, 0), (0, 1)):
        for color in (0, 4, 5, 14):
            w[color, 2 + dy, 2 + dx] = 1
    p["b1"][4] = -4

    p["W1"][6, 16, 2, 2] = 1  # row
    p["W1"][7, 17, 2, 2] = 1  # column

    # Selected row and column.
    for out, coord in ((0, 6), (1, 7)):
        p["W2"][out, 0, 1, 1] = 2
        p["W2"][out, coord, 1, 1] = 1
        p["b2"][out] = -2

    # Pack a goal's column and row. The global maximum picks the
    # rightmost goal, breaking column ties by row.
    p["W2"][2, 1, 1, 1] = 2
    p["W2"][2, 7, 1, 1] = 1
    p["W2"][2, 6, 1, 1] = 1 / 128
    p["b2"][2] = -2

    # Unfinished unselected piece and selected-on-goal indicators.
    p["W2"][3, 4, 1, 1] = 1
    p["W2"][3, 3, 1, 1] = -1
    p["W2"][4, 2, 1, 1] = 1

    # Unselected piece column and row for detecting pushes.
    for out, coord in ((5, 7), (6, 6)):
        p["W2"][out, 4, 1, 1] = 2
        p["W2"][out, coord, 1, 1] = 1
        p["b2"][out] = -2

    for i in range(7):
        p["W3"][i, i] = 1

    # Global maxima of the selected coordinates, packed goal, and
    # unselected coordinates.
    for i, f3_channel in enumerate((0, 1, 2, 5, 6)):
        p["Wz"][i, 32 + f3_channel] = 1

    # Both a completed selected goal and an unfinished other piece.
    p["Wz"][5, 32 + 3] = 1
    p["Wz"][5, 32 + 4] = 1
    p["bz"][5] = -1

    # Decode the column from the packed goal using ramp pairs.
    ramps = []
    for k in range(3, 62):
        i = 6 + 2 * (k - 3)
        left = (k - 1 + 63 / 128) / 63
        right = k / 63
        p["Wz"][i, 34] = 1
        p["bz"][i] = -left
        p["Wz"][i + 1, 34] = 1
        p["bz"][i + 1] = -right
        ramps.append((i, i + 1))

    wo = p["Wo"]
    bo = p["bo"]
    bo[:] = -10000
    bo[1:5] = 0

    # Signed goal-minus-piece coordinate differences.
    wo[4, 1] = -1
    wo[3, 1] = 1
    bo[4] = 2 / 63
    bo[3] = -2 / 63

    wo[2, 0] = -1
    wo[1, 0] = 1
    wo[2, 2] = 128
    wo[1, 2] = -128
    bo[2] = -256 / 63
    bo[1] = 256 / 63

    slope = 128 / 65
    for positive, negative in ramps:
        for idx, sign in ((positive, 1), (negative, -1)):
            s = sign * slope
            wo[4, idx] += s
            wo[3, idx] -= s
            wo[2, idx] -= 128 * s
            wo[1, idx] += 128 * s

    # Store selected row/column and unselected column/row.
    for unit, z_index in enumerate((0, 1, 3, 4)):
        p["Wmz"][unit, z_index] = 0.02

    threshold = 1.5 / 63

    # Detect movement of an unselected piece after a directional
    # action, which usually indicates a push.
    for i, (z_index, old_mem, sign) in enumerate((
        (3, 2, 1), (3, 2, -1),
        (4, 3, 1), (4, 3, -1),
    )):
        unit = 4 + i
        p["Wmz"][unit, z_index] = 100 * sign
        p["Wm"][unit, old_mem] = -5000 * sign
        for action in (1, 2, 3, 4):
            p["Wma"][unit, action] = 100
        p["bm"][unit] = -100 * (1 + threshold)

    # Detect a directional action that did not move the selected
    # piece, and reduce the score for repeating it.
    for i, (z_index, old_mem, sign) in enumerate((
        (0, 0, 1), (0, 0, -1),
        (1, 1, 1), (1, 1, -1),
    )):
        unit = 8 + i
        p["Wmz"][unit, z_index] = 100 * sign
        p["Wm"][unit, old_mem] = -5000 * sign
        p["Wma"][unit, 1 + i] = 100
        p["bm"][unit] = 100 * (threshold - 1)
        wo[1 + i, 128 + unit] = -0.5

    # Click after a push or after placing the selected piece.
    bo[6] = 6
    wo[6, 128 + 4:128 + 8] = 2
    wo[6, 5] = 5

    # Click the first unfinished unselected piece.
    p["Wc"][0, 3] = 1

    return p
