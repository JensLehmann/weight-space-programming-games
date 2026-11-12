import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,),
        "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,),
        "Wh": (16, 16, 3, 3), "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,),
        "Wm": (64, 64), "Wmz": (64, 128),
        "Wma": (64, 8), "bm": (64,),
        "Wo": (8, 192), "bo": (8,),
        "Wc": (1, 48), "bc": (1,),
    }
    p = {k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()}

    def c1(o, c, dy=0, dx=0, w=1.0):
        p["W1"][o, c, dy + 2, dx + 2] += w

    def c2(o, c, dy=0, dx=0, w=1.0):
        p["W2"][o, c, dy + 1, dx + 1] += w

    def f3(o, c, w=1.0):
        p["W3"][o, c] += w

    def rec(o, c, dy=0, dx=0, w=1.0):
        p["Wh"][o, c, dy + 1, dx + 1] += w

    dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    # Convolution offsets below are (row, column).
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Small- and large-footprint free-space detectors.
    floor = (2, 9, 10, 11, 14, 15)
    for out, offsets in [(0, (-1, 0)), (1, (-1, 0, 1, 2))]:
        for dy in offsets:
            for dx in offsets:
                for color in floor:
                    c1(out, color, dy, dx)
        p["b1"][out] = 1 - len(offsets) ** 2

    # Player location and orientation, for both sizes.
    base = np.array([[10, 10], [9, 9]], dtype=np.int8)
    rotations = (2, 0, 3, 1)
    rear = [
        [(1, -1), (1, 0)],
        [(-2, -1), (-2, 0)],
        [(-1, 1), (0, 1)],
        [(-1, -2), (0, -2)],
    ]
    for d, rotation in enumerate(rotations):
        small = np.rot90(base, rotation)
        for y in range(2):
            for x in range(2):
                c1(2 + d, int(small[y, x]), y - 1, x - 1)
        for dy, dx in rear[d]:
            c1(2 + d, 9, dy, dx, -4)
            c1(2 + d, 10, dy, dx, -4)
        p["b1"][2 + d] = -3

        large = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
        for y in range(4):
            for x in range(4):
                c1(6 + d, int(large[y, x]), y - 1, x - 1)
        p["b1"][6 + d] = -15

    # Exit interior corners.
    c1(10, 10, w=4)
    for dy, dx in dirs:
        c1(10, 9, dy, dx)
    p["b1"][10] = -5

    # Key core pixels, shifted by (+1,+1).
    # Each core pixel has exactly five color-6 neighbors.
    for color in (12, 13):
        c1(11, color, -1, -1, 8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                c1(11, 6, dy - 1, dx - 1)
    p["b1"][11] = -12

    # Centers of spell-panel cells.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            for color in (0, 2, 6, 11, 14, 15):
                c1(12, color, dy, dx)
    for dy, dx in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        c1(12, 3, dy, dx)
    p["b1"][12] = -12

    for color in (6, 11, 14, 15):
        c1(13, color)

    c1(14, 16)
    c1(15, 17)

    # Coordinate hinges partition the fixed nine-cell input panel.
    for out, threshold in zip(range(16, 20), (27, 28, 32, 33)):
        c1(out, 17, w=63)
        p["b1"][out] = -threshold
    for out, threshold in zip(range(20, 24), (52, 53, 57, 58)):
        c1(out, 16, w=63)
        p["b1"][out] = -threshold

    # Small teleport destination.
    for dy in (-1, 0):
        for dx in (-1, 0):
            c1(24, 15, dy, dx)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            c1(24, 3, dy, dx, -4)
    p["b1"][24] = -3

    # Large teleport destination.
    for dy in (-1, 2):
        for dx in (-1, 2):
            c1(25, 11, dy, dx)
    for dy in (0, 1):
        for dx in (0, 1):
            c1(25, 15, dy, dx, -4)
    p["b1"][25] = -3

    # Spell availability icons.
    for out, color in [(26, 15), (27, 11)]:
        for dy in (0, 1):
            for dx in (0, 1):
                c1(out, color, dy, dx)
        for dy in (0, 1):
            c1(out, 3, dy, -1)
        p["b1"][out] = -5

    for dy in (0, 1):
        for dx in (0, 1):
            c1(28, 6, dy, dx)
    for dx in (0, 1):
        c1(28, 3, -1, dx)
    p["b1"][28] = -5

    # Key-border seeds for the navigation field.
    c1(29, 6, w=9)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                for color in (12, 13):
                    c1(29, color, dy, dx)
    p["b1"][29] = -9

    # Aggregate player detectors.
    face2 = (2, 29, 30, 31)
    for d in range(4):
        c2(0, 2 + d)
        c2(1, 6 + d)
        c2(face2[d], 2 + d)
        c2(face2[d], 6 + d)

    c2(3, 0)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            c2(4, 10, dy, dx)
            c2(4, 29, dy, dx)

    # Four nonoverlapping shifts turn a 2x2 core detector into a
    # binary 4x4 firing-alignment rectangle.
    for dy in (-1, 1):
        for dx in (-1, 1):
            c2(5, 11, dy, dx)
            c2(6, 11, dy, dx, 2)
            c2(7, 11, dy, dx, 2)
    c2(6, 15)
    c2(7, 14)
    p["b2"][6] = -2 - 3 / 63
    p["b2"][7] = -2 - 3 / 63

    c2(8, 15)
    c2(9, 14)
    c2(10, 12)
    c2(11, 12)
    c2(11, 13)
    p["b2"][11] = -1

    x_regions = [
        (1, {16: -1, 17: 1}),
        (0, {16: 1, 17: -1, 18: -1, 19: 1}),
        (0, {18: 1, 19: -1}),
    ]
    y_regions = [
        (1, {20: -1, 21: 1}),
        (0, {20: 1, 21: -1, 22: -1, 23: 1}),
        (0, {22: 1, 23: -1}),
    ]
    for row in range(3):
        for col in range(3):
            out = 12 + 3 * row + col
            c2(out, 12)
            xc, xt = x_regions[col]
            yc, yt = y_regions[row]
            for channel, weight in xt.items():
                c2(out, channel, w=weight)
            for channel, weight in yt.items():
                c2(out, channel, w=weight)
            p["b2"][out] = xc + yc - 2

    for d, (dy, dx) in enumerate(dirs):
        c2(21 + d, 1, dy, dx)

    c2(25, 25)
    c2(26, 24)
    c2(27, 26)
    c2(28, 27)

    # Axial features and panel XORs.
    face3 = (2, 28, 30, 31)
    for out, inp in [
        (0, 0), (1, 1), (3, 3), (4, 4),
        (5, 25), (6, 26), (7, 10), (8, 27), (29, 28),
    ]:
        f3(out, inp)
    for out, inp in zip(face3, face2):
        f3(out, inp)

    for d in range(4):
        f3(9 + d, 1)
        f3(9 + d, 21 + d, -1)

    for d in range(4):
        vertical = d < 2
        coord = 9 if vertical else 8
        key_coord = 64 + 7 if vertical else 32 + 6
        key_flag = 64 + 5 if vertical else 32 + 5
        sign = 1 if d in (0, 2) else -1
        for high in (0, 1):
            out = 13 + 2 * d + high
            f3(out, coord, 63 * sign)
            f3(out, key_coord, -63 * sign)
            f3(out, 0, 256)
            f3(out, 1, 256)
            f3(out, key_flag, 256)
            p["b3"][out] = -512 - high

    patterns = [
        (0, 1, 4),       # teleport
        (1, 4, 7),       # fire
        (1, 3, 5, 7),    # resize
    ]
    for spell, slots in enumerate(patterns):
        missing = 21 + 2 * spell
        extra = missing + 1
        for slot in slots:
            f3(missing, 12 + slot)
            f3(extra, 12 + slot, -1)
        f3(missing, 11, -1)
        f3(extra, 11)

    f3(27, 10, -1)
    p["b3"][27] = 1

    # Recurrent navigation and communication.
    B = float(2 ** 44)
    S = float(2 ** 24)

    # Capped reachability represented by the difference of two ReLUs.
    for out in (0, 1):
        for dy, dx in [(0, 0)] + dirs:
            rec(out, 0, dy, dx)
            rec(out, 1, dy, dx, -1)
        p["Uh"][out, 3] = 8
        p["Uh"][out, 4] = 9
        p["bh"][out] = -8 - out

    rec(2, 2)
    rec(2, 0)
    rec(2, 1, w=-1)

    def player_gate(out):
        p["Uh"][out, 0] += B
        p["Uh"][out, 1] += B
        p["bh"][out] -= B

    for d, (dy, dx) in enumerate(dirs):
        rec(3 + d, 2, dy, dx)
        player_gate(3 + d)

    rec(15, 2)
    player_gate(15)

    for out, marker in [(7, 6), (8, 5)]:
        rec(out, 2)
        p["Uh"][out, marker] = B
        p["bh"][out] = -B

    def diffuse(channel):
        # Total gain 9/8, rather than 1.44. Over 128 iterations this
        # leaves substantial float32 headroom.
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                rec(channel, channel, dy, dx, 0.125)

    diffuse(9)
    diffuse(10)
    rec(9, 7, w=1 / 128)
    rec(10, 8, w=1 / 128)

    def panel_mask(out, spell):
        p["Uh"][out, 7] -= B
        p["Uh"][out, 21 + 2 * spell] += B
        p["Uh"][out, 22 + 2 * spell] += B

    # Fire request field.
    diffuse(11)
    rec(11, 15, w=1 / 128)
    p["Uh"][11, 0] -= 1
    p["Uh"][11, 1] -= 1
    for d in range(4):
        p["Uh"][11, 13 + 2 * d] += 1
        p["Uh"][11, 14 + 2 * d] -= 1
    panel_mask(11, 1)

    # Grow request at a stranded small player.
    rec(12, 10)
    rec(12, 9, w=-1)
    for d in range(4):
        rec(12, 3 + d, w=-S)
    p["Uh"][12, 0] = B
    p["bh"][12] = -B

    # Shrink request: reachable small-footprint route or small portal.
    rec(13, 9)
    rec(13, 10, w=-1)
    for d in range(4):
        rec(13, 3 + d, w=S)
    p["Uh"][13, 1] = B
    p["bh"][13] = -B

    diffuse(14)
    rec(14, 12, w=1 / S)
    rec(14, 13, w=1 / S)
    panel_mask(14, 2)

    # Global action readout.
    next_unit = 0

    def unit(coefficients, bias, outputs):
        nonlocal next_unit
        j = next_unit
        next_unit += 1
        for index, weight in coefficients.items():
            p["Wz"][j, index] += weight
        p["bz"][j] = bias
        for action, weight in outputs.items():
            p["Wo"][action, j] += weight

    def fm(channel):
        return 32 + channel

    for d in range(4):
        unit(
            {3 + d: 1, 15: -1, fm(9 + d): -256},
            1,
            {1 + d: 1},
        )

        unit(
            {
                3 + d: 1, 15: -1,
                fm(9 + d): 256, fm(8): 256,
            },
            0.25 - 512,
            {6: 3},
        )

        lo = fm(13 + 2 * d)
        hi = fm(14 + 2 * d)
        aim = {15: 4, lo: 260, hi: -256}

        # Permit a short movement-only detour around a firing grate.
        threshold = 125.5
        for offset, weight in [(0, 20), (-1, -20)]:
            unit(
                aim,
                -4 * threshold - 256 + offset,
                {1 + d: weight},
            )

        shoot = dict(aim)
        shoot[fm(face3[d])] = 256
        for offset, weight in [(0, 26), (-1, -26)]:
            unit(
                shoot,
                -4 * threshold - 512 + offset,
                {6: weight},
            )

    unit(
        {7: 1, 8: 1, 15: -512, fm(29): 512},
        -512,
        {6: 0.4},
    )

    # Record whether the actor has issued an orientation action.
    p["Wm"][0, 0] = 4
    p["Wma"][0, 1:5] = 5
    p["Wo"][6, 128] = 6
    p["bo"][6] = -6
    for action in (0, 5, 7):
        p["bo"][action] = -10000

    # Panel-only clicks. Fire overrides resizing; teleport is the
    # default intervention when no local route exists.
    p["Wc"][0, 27] = -float(2 ** 70)
    p["Wc"][0, 32 + 11] = float(2 ** 20)
    p["Wc"][0, 32 + 14] = 1
    p["Wc"][0, 21] = float(2.0 ** -120)
    p["Wc"][0, 22] = float(2.0 ** -120)

    return p
