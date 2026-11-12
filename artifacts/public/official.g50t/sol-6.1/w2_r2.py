import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,),
        "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,),
        "Wh": (16, 16, 3, 3),
        "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,),
        "Wm": (64, 64), "Wmz": (64, 128),
        "Wma": (64, 8), "bm": (64,),
        "Wo": (8, 192), "bo": (8,),
        "Wc": (1, 48), "bc": (1,),
    }
    p = {k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()}
    W1, W2, W3 = p["W1"], p["W2"], p["W3"]
    b1, b2, b3 = p["b1"], p["b2"], p["b3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def first(out, channel, dy=0, dx=0, weight=1.0):
        W1[out, channel, dy + 2, dx + 2] += weight

    def second(out, channel, dy=0, dx=0, weight=1.0):
        W2[out, channel, dy + 1, dx + 1] += weight

    def recurrent(out, channel, dy=0, dx=0, weight=1.0):
        Wh[out, channel, dy + 1, dx + 1] += weight

    # Player and ghost centres.
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            centre = dy == 0 and dx == 0
            first(0, 5 if centre else 9, dy, dx)
            first(26, 5 if centre else 2, dy, dx)
    b1[0] = -8.0
    b1[26] = -8.0

    # Eroded terrain and inflated gate-body detection.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            for colour in range(1, 16):
                first(1, colour, dy, dx)
            for colour in (8, 11):
                first(3, colour, dy, dx)
    b1[1] = -24.0
    b1[3] = -12.0

    # Exit centre.
    first(2, 9, weight=9.0)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy or dx:
                first(2, 5, dy, dx)
    b1[2] = -16.0

    def plate_detector(out, colours):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                for colour in colours:
                    first(out, colour, dy, dx)
        for dy, dx in directions:
            for colour in colours:
                first(out, colour, 2 * dy, 2 * dx, -3.0)
        b1[out] = -5.5

    plate_detector(4, (8, 11, 15))
    plate_detector(5, (8, 15))

    # Displaced samples recognize a wire outside a player's body.
    for k, (dy, dx) in enumerate(directions):
        for colour in (8, 15):
            first(12 + k, colour, 2 * dy, 2 * dx)

    first(16, 16)
    first(17, 17)

    # Shifted player markers.
    for k, (dy, dx) in enumerate(directions):
        out = 19 + k
        for sy in range(-1, 2):
            for sx in range(-1, 2):
                colour = 5 if (sy == 0 and sx == 0) else 9
                first(out, colour, sy - dy, sx - dx)
        b1[out] = -8.0

    second(0, 0)
    second(1, 1)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            second(1, 3, dy, dx, -10.0)

    second(2, 2)
    second(3, 4, weight=2.0)
    second(4, 5, weight=2.0)
    second(23, 1)
    second(26, 26)

    # Wire contact for the player and for ghosts.
    for out, marker in ((11, 0), (27, 26)):
        second(out, marker, weight=4.0)
        for k, (dy, dx) in enumerate(directions):
            second(out, 12 + k, dy, dx)
        b2[out] = -4.0

    # Sparse plate-coordinate channels, subsequently pooled axially.
    for out, coordinate, sign, offset in (
        (12, 17, 1.0, 0.0),
        (13, 17, -1.0, 1.0),
        (14, 16, 1.0, 0.0),
        (15, 16, -1.0, 1.0),
    ):
        second(out, 5, weight=256.0)
        second(out, coordinate, weight=sign)
        b2[out] = offset - 128.0

    second(16, 16)
    second(17, 17)

    for k, (dy, dx) in enumerate(directions):
        second(19 + k, 19 + k, -dy, -dx)
        second(28 + k, 1, dy, dx)
        second(28 + k, 3, dy, dx, -10.0)

    # f3 masks.
    for out, channel in (
        (0, 0), (1, 1), (2, 2), (3, 3),
        (4, 23), (5, 11), (29, 26), (30, 27),
    ):
        W3[out, channel] = 1.0

    for k in range(4):
        W3[19 + k, 19 + k] = 1.0
        # Invalid-direction marker.
        W3[25 + k, 19 + k] = 1.0
        W3[25 + k, 28 + k] = -1.0

    # Three hinge functions form an exact triangular next-cell test.
    guard = 512.0
    for direction in range(4):
        for j, threshold in enumerate((5.0, 6.0, 7.0)):
            out = 7 + 3 * direction + j
            W3[out, 0] = guard
            b3[out] = -2.0 * guard - threshold

            if direction == 0:
                W3[out, 16] = 63.0
                W3[out, 64 + 15] = 63.0
                W3[out, 64 + 4] = guard
                b3[out] -= 63.0
            elif direction == 1:
                W3[out, 16] = -63.0
                W3[out, 64 + 14] = 63.0
                W3[out, 64 + 4] = guard
            elif direction == 2:
                W3[out, 17] = 63.0
                W3[out, 32 + 13] = 63.0
                W3[out, 32 + 4] = guard
                b3[out] -= 63.0
            else:
                W3[out, 17] = -63.0
                W3[out, 32 + 12] = 63.0
                W3[out, 32 + 4] = guard

    # Three bounded flood fills:
    # 0:2 exit on current terrain,
    # 2:4 switches on current terrain,
    # 4:6 exit on terrain with gates omitted.
    for lo, hi, terrain, seed in (
        (0, 1, 1, 2),
        (2, 3, 1, 3),
        (4, 5, 4, 2),
    ):
        for out in (lo, hi):
            for dy, dx in directions:
                recurrent(out, lo, dy, dx)
                recurrent(out, hi, dy, dx, -1.0)
            Uh[out, terrain] = 5.0
            Uh[out, seed] = 1.0
            bh[out] = -5.0 - float(out == hi)

    # First-arrival counters encode route costs.
    recurrent(6, 6)
    recurrent(6, 2, weight=-1.0)
    recurrent(6, 3)
    bh[6] = 1.0

    recurrent(7, 7)
    recurrent(7, 4, weight=-1.0)
    recurrent(7, 5)
    bh[7] = 1.0

    # Directional exit and switch costs, sampled ahead of the player.
    for k, (dy, dx) in enumerate(directions):
        out = 8 + k
        recurrent(out, 7, dy, dx)
        recurrent(out, 0, dy, dx, -200.0)
        recurrent(out, 1, dy, dx, 200.0)
        Uh[out, 19 + k] = 1000.0
        Uh[out, 25 + k] = 400.0
        bh[out] = -800.0

        out = 12 + k
        recurrent(out, 6, dy, dx)
        Uh[out, 19 + k] = 1000.0
        Uh[out, 25 + k] = 400.0
        bh[out] = -1000.0

    Wz = p["Wz"]
    for k in range(4):
        Wz[k, 8 + k] = 1.0

        # Correction selects the cheaper exit or switch objective.
        Wz[4 + k, 8 + k] = 1.0
        Wz[4 + k, 12 + k] = -1.0
        p["bz"][4 + k] = -18.0

        base = 32 + 7 + 3 * k
        Wz[8 + k, base] = 1.0
        Wz[8 + k, base + 1] = -2.0
        Wz[8 + k, base + 2] = 1.0

    Wz[12, 32 + 5] = 1.0

    # g = [h.max(16), h.mean(16), f3.max(32), f3.mean(32)].
    # Thus f3 means begin at index 64, not 80.
    Wz[14, 64 + 29] = 4096.0
    Wz[14, 64 + 30] = -4096.0

    for k in range(4):
        # A moving ghost selects the exit-gradient branch until it
        # reaches its recorded switch.
        p["Wmz"][k, 4 + k] = 0.0003
        p["Wmz"][k, 14] = 30.0
        p["Wma"][k, 5] = 30.0

        # Store an approaching momentary plate.
        p["Wmz"][4 + k, 8 + k] = 10.0
        p["bm"][4 + k] = -5.0

        # Confirm arrival using the previous action.
        p["Wm"][8 + k, 4 + k] = 5.0
        p["Wma"][8 + k, 1 + k] = 10.0
        p["bm"][8 + k] = -10.0

    p["bo"][:] = -1.0e6
    for k in range(4):
        action = 1 + k
        p["bo"][action] = 0.0
        p["Wo"][action, k] = -1.0
        p["Wo"][action, 128 + k] = 1.0 / 0.0003

    # Record only after entering a momentary plate.
    p["bo"][5] = 1000.0
    p["Wo"][5, 12] = 2000.0
    for k in range(4):
        p["Wo"][5, 128 + 8 + k] = 1000.0

    return p
