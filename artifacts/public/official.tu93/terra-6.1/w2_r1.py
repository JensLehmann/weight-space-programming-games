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
    p = {k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()}

    W1, b1 = p["W1"], p["b1"]
    W2, b2 = p["W2"], p["b2"]
    W3 = p["W3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Player center.
    W1[0, 9, 2, 2] = 1
    for dy, dx in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        W1[0, 9, 2 + dy, 2 + dx] = 1
    for dy, dx in directions:
        W1[0, 4, 2 + dy, 2 + dx] = 1
    b1[0] = -5

    # Corridor probes.
    for i, (dy, dx) in enumerate(directions):
        W1[1 + i, 2, 2 + 2 * dy, 2 + 2 * dx] = 1

    W1[5, 14, 2, 2] = 1
    for color in range(16):
        if color not in (5, 6):
            W1[6, color, 2, 2] = 1
    for color in (0, 8, 11, 12, 13, 14, 15):
        W1[7, color, 2, 2] = 1

    # Directional enemy markers, positioned two pixels ahead of center.
    for i, (dy, dx) in enumerate(directions):
        my, mx = 2 - dy, 2 - dx
        by, bx = 2 - 2 * dy, 2 - 2 * dx

        # Blue and pink.
        c = 8 + i
        for marker in (11, 15):
            W1[c, marker, my, mx] = 2
        for body in (8, 12):
            W1[c, body, by, bx] = 2
            W1[c, body, 2, 2] = -2
        b1[c] = -3

        # Purple, whether active or idle.
        c = 12 + i
        for marker in (11, 15):
            W1[c, marker, my, mx] = 2
        W1[c, 13, by, bx] = 2
        W1[c, 13, 2, 2] = -2
        b1[c] = -3

    # Legal-action indicators at the player center.
    for i, (dy, dx) in enumerate(directions):
        W2[i, 0, 1, 1] = 1
        W2[i, 1 + i, 1 + dy, 1 + dx] = 1
        b2[i] = -1

    W2[4, 6, 1, 1] = 1  # Traversable.
    W2[5, 5, 1, 1] = 1  # Exit.
    W2[6, 7, 1, 1] = 1  # Node pixels.

    # Widen enemy projection sources to the width of a node.
    for i, (dy, dx) in enumerate(directions):
        for lateral in (-1, 0, 1):
            oy = -dy + lateral * dx
            ox = -dx - lateral * dy
            W2[7 + i, 8 + i, 1 + oy, 1 + ox] = 1
            W2[11 + i, 12 + i, 1 + oy, 1 + ox] = 1

    for c in range(15):
        W3[c, c] = 1

    # Saturated exit wave. The difference h[0]-h[1] is the reached mask.
    for out in (0, 1):
        for dy, dx in [(0, 0)] + directions:
            Wh[out, 0, 1 + dy, 1 + dx] += 1
            Wh[out, 1, 1 + dy, 1 + dx] -= 1
        Uh[out, 4] = 8
        Uh[out, 5] = 1
        for enemy in range(8, 12):
            Wh[out, enemy, 1, 1] = -144
    bh[0] = -8
    bh[1] = -9

    # Accumulated arrival field, larger closer to the exit.
    Wh[2, 2, 1, 1] = 1
    Wh[2, 0, 1, 1] = 1
    Wh[2, 1, 1, 1] = -1

    # Player beams along each legal outgoing direction.
    for i, (dy, dx) in enumerate(directions):
        c = 4 + i
        Uh[c, i] = 1
        Wh[c, c, 1 - dy, 1 - dx] = np.float32(0.5)

    # Enemy beams. Purple receives a longer effective projection.
    for i, (dy, dx) in enumerate(directions):
        c = 8 + i
        Uh[c, 7 + i] = 1
        Uh[c, 11 + i] = 32
        Wh[c, c, 1 - dy, 1 - dx] = np.float32(0.5)

    # Read the field at the near edge of each next node.
    for i in range(4):
        c = 12 + i
        Wh[c, 2, 1, 1] = 1
        Wh[c, 4 + i, 1, 1] = 8192
        Uh[c, 6] = 16384
        bh[c] = -16640
        for enemy in range(8, 12):
            Wh[c, enemy, 1, 1] = -4096

    # Scores and legal-action baseline.
    for i in range(4):
        p["Wz"][i, 12 + i] = 1
        p["Wz"][4 + i, 32 + i] = 1
        p["Wo"][1 + i, i] = 1
        p["Wo"][1 + i, 4 + i] = 200

    # Small immediate-reversal penalty.
    for i in range(4):
        p["Wma"][i, 1 + i] = 3
    opposite = [1, 0, 3, 2]
    for i in range(4):
        p["Wo"][1 + i, 128 + opposite[i]] = -1

    return p
