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
    W2, W3 = p["W2"], p["W3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    # Directions in action order: up, down, left, right.
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Traversable pixels, exit pixels, and the bottom UI row.
    for color in range(16):
        if color not in (5, 6):
            W1[0, color, 2, 2] = 1
    W1[1, 14, 2, 2] = 1
    W1[2, 16, 2, 2] = 63
    b1[2] = -62

    # Legal-action probes outside each side of the player.
    for i, (dr, dc) in enumerate(dirs):
        ch = 3 + i
        if dr:
            offsets = [(-2 * dr, c) for c in (-1, 0, 1)]
        else:
            offsets = [(r, -2 * dc) for r in (-1, 0, 1)]
        for r, c in offsets:
            W1[ch, 9, r + 2, c + 2] = 1
            W1[ch, 4, r + 2, c + 2] = 1
        W1[ch, 2, 2, 2] = 1
        b1[ch] = -3.5

    # Oriented enemy detectors.
    for i, eye in enumerate(dirs):
        for active13 in (False, True):
            ch = (11 if active13 else 7) + i
            for r in (-1, 0, 1):
                for c in (-1, 0, 1):
                    if (r, c) == eye:
                        W1[ch, 11 if active13 else 15,
                           r + 2, c + 2] = 1
                    elif active13:
                        W1[ch, 13, r + 2, c + 2] = 1
                    else:
                        W1[ch, 8, r + 2, c + 2] = 1
                        W1[ch, 12, r + 2, c + 2] = 1
            b1[ch] = -8.5

    # Features for detecting non-goal dead-end nodes. The second
    # convolution completes the three-pixel neighbor offsets.
    for i, (dr, dc) in enumerate(dirs):
        W1[15 + i, 2, 2 + 2 * dr, 2 + 2 * dc] = 1
    for color in (0, 4, 8, 9, 11, 12, 13, 14, 15):
        W1[19, color, 2, 2] = 1

    # Remaining-step bar.
    W1[20, 6, 2, 2] = 1

    for ch in range(3):
        W2[ch, ch, 1, 1] = 1

    # Move probes one pixel closer to their candidate destinations.
    for i, (dr, dc) in enumerate(dirs):
        W2[3 + i, 3 + i, 1 - dr, 1 - dc] = 2

    # Dilate detected enemy centers to their bodies.
    for i in range(4):
        W2[7 + i, 7 + i, :, :] = 2
        W2[7 + i, 11 + i, :, :] = 2

    W2[15, 19, 1, 1] = 1
    for i, (dr, dc) in enumerate(dirs):
        W2[16, 15 + i, 1 + dr, 1 + dc] = 1
    W2[20, 20, 1, 1] = 1

    # f3: path, goal, enemy bodies, action probes, oriented bodies.
    W3[0, 0] = 1
    W3[0, 2] = -1
    W3[1, 1] = 1
    W3[1, 2] = -1
    for i in range(4):
        W3[2, 7 + i] = 1
        W3[3 + i, 3 + i] = 1
        W3[7 + i, 7 + i] = 1

    # One outgoing corridor marks a dead end; exempt the exit.
    W3[11, 15] = 2
    W3[11, 16] = -1
    W3[11, 1] = -2
    W3[11, 2] = -2
    W3[12, 20] = 1

    # Saturating flood from the exit: h[0] - h[1] is the
    # approximately binary reached mask.
    for out in (0, 1):
        for dr, dc in dirs:
            Wh[out, 0, 1 + dr, 1 + dc] = 1
            Wh[out, 1, 1 + dr, 1 + dc] = -1
        Wh[out, 15, 1, 1] = -8
        Uh[out, 0] = 5
        Uh[out, 1] = 1
    bh[0] = -5
    bh[1] = -6

    # Decaying rays in enemy-facing directions.
    for i, (dr, dc) in enumerate(dirs):
        ch = 6 + i
        Wh[ch, ch, 1 - dr, 1 - dc] = 0.5
        Uh[ch, 7 + i] = 1

    # Threshold differences isolate the square ahead of an enemy.
    for out, threshold in zip((10, 11, 12, 13), (1, 2, 16, 17)):
        for source in range(6, 10):
            Wh[out, source, 1, 1] = 128
        Uh[out, 2] = -4096
        bh[out] = -threshold

    # Clock. At dead ends it grows faster, providing a local cost.
    Wh[14, 14, 1, 1] = 1
    Uh[14, 11] = 1
    bh[14] = 1

    # Initially delay the flood while enemy rays settle; release
    # threatened squares later so they do not block routes forever.
    Wh[15, 10, 1, 1] = 20
    Wh[15, 11, 1, 1] = -20
    Wh[15, 12, 1, 1] = -20
    Wh[15, 13, 1, 1] = 20
    Wh[15, 14, 1, 1] = -1
    bh[15] = 9

    # Accumulate arrival time, immediate threat, and dead-end cost
    # at each legal action probe.
    for i, (dr, dc) in enumerate(dirs):
        ch = 2 + i
        Wh[ch, ch, 1, 1] = 1
        Wh[ch, 0, 1 + dr, 1 + dc] = -1
        Wh[ch, 1, 1 + dr, 1 + dc] = 1
        Wh[ch, 15, 1 + dr, 1 + dc] = 0.4
        Wh[ch, 14, 1 + dr, 1 + dc] = 0.0018
        Wh[ch, 14, 1, 1] = -0.0018
        Uh[ch, 3 + i] = 129
        bh[ch] = -128

    for i in range(4):
        p["Wz"][i, 2 + i] = 1
        p["Wz"][4 + i, 32 + 3 + i] = 1

    # The mean of f3 channel 12 is the number of lit bar pixels
    # divided by 4096.
    p["Wz"][8, 64 + 12] = -4096.0 / 14.0
    p["bz"][8] = 1

    # Previous action for a small reversal penalty.
    for i in range(4):
        p["Wma"][i, 1 + i] = 1

    opposite = [1, 0, 3, 2]
    p["bo"][:] = -2048
    for i in range(4):
        action = 1 + i
        p["bo"][action] = 0
        p["Wo"][action, i] = -1
        p["Wo"][action, 4 + i] = 2048
        p["Wo"][action, 128 + i] = 0.2
        p["Wo"][action, 128 + opposite[i]] = -3.5

    p["bo"][0] = 1600
    p["Wo"][0, 8] = 3000

    return p
