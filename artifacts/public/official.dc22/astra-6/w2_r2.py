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
    p = {k: np.zeros(shape, dtype=np.float32)
         for k, shape in shapes.items()}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    actions = [1, 2, 3, 4]
    opposite = [1, 0, 3, 2]

    def c1(o, i, dy=0, dx=0, w=1):
        p["W1"][o, i, dy + 2, dx + 2] += w

    def c2(o, i, dy=0, dx=0, w=1):
        p["W2"][o, i, dy + 1, dx + 1] += w

    def ch(o, i, dy=0, dx=0, w=1):
        p["Wh"][o, i, dy + 1, dx + 1] += w

    def f3(o, i, w=1):
        p["W3"][o, i] += w

    # Upper-left pixel of the player.
    c1(0, 14)
    c1(0, 14, -1, 0, -1)
    c1(0, 14, 0, -1, -1)

    # Fully painted support footprints.
    for colour in (1, 2, 6, 7, 8, 9, 11, 12, 13, 14, 15):
        for dy in (0, 1):
            for dx in (0, 1):
                c1(1, colour, dy, dx)
    p["b1"][1] = -3

    # Half-counts distinguish solid and checkerboard platforms.
    for out, colour in ((2, 9), (3, 15)):
        for dy in (0, 1):
            for dx in (0, 1):
                c1(out, colour, dy, dx, 0.5)

    # Teleporter endpoints.
    for out, pattern in (
        (4, ((7, 6), (6, 7))),
        (5, ((6, 7), (7, 6))),
    ):
        for dy in range(2):
            for dx in range(2):
                c1(out, pattern[dy][dx], dy, dx)
        p["b1"][out] = -3

    # Isolated goal square.
    c1(6, 11)
    c1(6, 11, 1, 1)
    for dy, dx in ((-1, 0), (0, -1), (2, 0), (0, 2)):
        c1(6, 11, dy, dx, -1)
    p["b1"][6] = -1

    # Collectible control-unlock icon.
    for dx in (0, 1):
        c1(7, 15, 0, dx)
    for dx in (-1, 0, 1, 2):
        c1(7, 15, 1, dx)
    c1(7, 15, 0, -1, -1)
    p["b1"][7] = -5

    # Pivot indicators and coordinates.
    c1(8, 13)
    c1(9, 16, w=63)
    c1(9, 13, w=64)
    p["b1"][9] = -64
    c1(10, 17, w=63)
    c1(10, 13, w=64)
    p["b1"][10] = -64
    c1(11, 16, w=63)
    c1(12, 17, w=63)

    # Control-button recognition.
    for out, colour in zip(range(13, 17), (8, 9, 6, 15)):
        for dy in range(-2, 3):
            c1(out, colour, dy, 0)
        c1(out, 5, -2, -2)
        c1(out, 0, -2, -2)
        p["b1"][out] = -5

    c1(17, 17, w=-63)
    p["b1"][17] = 34

    # One-pixel-displaced player anchors.
    for j, (dy, dx) in enumerate(directions):
        c1(20 + j, 14, -dy, -dx)
        c1(20 + j, 14, -dy - 1, -dx, -1)
        c1(20 + j, 14, -dy, -dx - 1, -1)

    def target(out, j, weight=1):
        dy, dx = directions[j]
        c2(out, 20 + j, -dy, -dx, weight)

    for j in range(4):
        c2(j, 20 + j)
        target(4, j)

    c2(5, 1)

    # Switch demand is restricted to two-pixel move destinations.
    for out, raw in ((6, 2), (7, 3)):
        for j in range(4):
            target(out, j, 8)
        c2(out, raw)
        c2(out, 1, w=-2)
        p["b2"][out] = -8

    for j in range(4):
        target(8, j)
    c2(8, 1, w=-1)
    c2(8, 2, w=-1)
    c2(8, 3, w=-1)

    c2(9, 4)
    c2(9, 5)
    for out, inp in (
        (10, 6), (11, 7), (12, 8), (13, 9), (14, 10),
        (15, 2), (16, 3), (17, 11), (18, 12),
    ):
        c2(out, inp)

    for j in range(4):
        c2(19 + j, 13 + j)
        c2(19 + j, 17, w=-1)

        target(23 + j, j)
        c2(23 + j, 1)
        p["b2"][23 + j] = -1

        target(28 + j, j)
        c2(28 + j, 4)
        c2(28 + j, 5)
        p["b2"][28 + j] = -1

    c2(27, 0)

    # Axial geometry and local movement status.
    for j in range(4):
        f3(j, j)

    f3(4, 5)
    f3(4, 15)
    # Brown checkerboards are deliberately NOT optimistic support.
    # Solid brown platforms are already included in channel 5.

    for out, inp in (
        (5, 6), (6, 7), (7, 8), (8, 9),
        (9, 10), (10, 11),
        (11, 19), (12, 20), (13, 21), (14, 22),
    ):
        f3(out, inp)

    for j in range(4):
        f3(15 + j, 23 + j)
        f3(15 + j, 28 + j, 2)
    f3(19, 27)

    # Difference-of-ReLU ramps bound potential pivot arms.
    def arm_ramps(start, coordinate, pivot_coordinate, axial_offset):
        pc = axial_offset + pivot_coordinate
        present = axial_offset + 12

        for offset, extra in ((0, 0), (1, -1)):
            out = start + offset
            f3(out, pc)
            f3(out, coordinate, -1)
            f3(out, present, 128)
            p["b3"][out] = -139.25 + extra

        for offset, extra in ((2, 0), (3, -1)):
            out = start + offset
            f3(out, coordinate)
            f3(out, pc, -1)
            f3(out, present, -128)
            p["b3"][out] = 120.75 + extra

    arm_ramps(20, 17, 13, 64)
    arm_ramps(24, 18, 14, 32)

    for out, coordinate, marker in (
        (28, 17, 27), (29, 18, 27),
        (30, 17, 10), (31, 18, 10),
    ):
        f3(out, coordinate)
        f3(out, marker, 64)
        p["b3"][out] = -64

    # Recurrent channel 0: obstacle penalty.
    p["bh"][0] = -1
    p["Uh"][0, 4] = -4
    for start in (20, 24):
        for k, weight in enumerate((1, -1, 1, -1)):
            p["Uh"][0, start + k] = weight

    # Channels 1,2: a clipped reachability wave.
    for out in (1, 2):
        for dy, dx in directions:
            ch(out, 1, dy, dx)
            ch(out, 2, dy, dx, -1)
        ch(out, 0, w=-16)
        ch(out, 4)
        p["Uh"][out, 9] = 1
        p["Uh"][out, 10] = 1
    p["bh"][2] = -1

    # Channel 3 integrates first-arrival time.
    ch(3, 3)
    ch(3, 1)
    ch(3, 2, w=-1)

    # Channels 4..6 relay reachability between visible teleporters.
    ch(4, 1)
    ch(4, 2, w=-1)
    ch(4, 5)
    ch(4, 6, w=-1)
    p["Uh"][4, 8] = 32
    p["bh"][4] = -32

    neighbours = [
        (dy, dx)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if dy or dx
    ]
    for out in (5, 6):
        for dy, dx in neighbours:
            ch(out, 5, dy, dx)
            ch(out, 6, dy, dx, -1)
        ch(out, 4)
    p["bh"][6] = -1

    # Channels 7..10: directional improvement.
    for j, (dy, dx) in enumerate(directions):
        ch(7 + j, 3, dy, dx)
        ch(7 + j, 3, -dy, -dx, -1)
        p["Uh"][7 + j, j] = 256
        p["bh"][7 + j] = -256

    signal = 1.0e8
    barrier = 1.0e16
    controls = (11, 12, 13, 14)
    needs = (5, 6, 7)

    # Channels 11..13 carry requests to matching switch buttons.
    for out, need, own_control, frontier in (
        (11, 7, 11, 0.0),
        (12, 5, 12, 0.5),
        (13, 6, 14, 0.5),
    ):
        for dy, dx in neighbours:
            ch(out, out, dy, dx, 0.12)
        for j, (dy, dx) in enumerate(directions):
            ch(out, 7 + j, -dy, -dx, signal)

        for feat in range(15, 19):
            p["Uh"][out, feat] -= 100000 * signal
        for feat in needs:
            if feat != need:
                p["Uh"][out, feat] -= 100000 * signal
        p["Uh"][out, need] += frontier * signal

        for feat in controls:
            if feat != own_control:
                p["Uh"][out, feat] -= barrier

    # Channel 14 provides the teleporter-control click target.
    for dy, dx in neighbours:
        ch(14, 14, dy, dx, 0.12)
    p["Uh"][14, 19] = 1
    for feat in controls:
        if feat != 13:
            p["Uh"][14, feat] = -barrier

    # Channel 15 restricts carrier values to button pixels.
    for inp in range(11, 15):
        ch(15, inp)
    p["bh"][15] = -barrier
    for feat in controls:
        p["Uh"][15, feat] = barrier

    p["Wc"][0, 47] = 1

    def fm(c):
        return 32 + c

    def fa(c):
        return 64 + c

    for j in range(4):
        p["Wz"][j, 7 + j] = 1

        p["Wz"][4 + j, fm(15 + j)] = -1
        p["bz"][4 + j] = 1

        p["Wz"][8 + j, fm(15 + j)] = 1
        p["bz"][8 + j] = -2

        p["Wz"][12 + j, 7 + j] = 1
        p["Wz"][12 + j, fm(15 + j)] = -256

    # Frontier requests require a currently visible matching control.
    for unit, need, control in ((16, 5, 12), (17, 6, 14)):
        p["Wz"][unit, fm(need)] = 1
        p["Wz"][unit, fm(control)] = 1
        p["bz"][unit] = -1

    p["Wz"][18, fa(8)] = -4096
    p["Wz"][18, fm(13)] = 256
    p["bz"][18] = -254

    for j in range(4):
        p["Wz"][20 + j, fm(28 + j)] = 1

    # Remember moves onto teleporters and the previous direction.
    for j, action in enumerate(actions):
        p["Wmz"][j, 8 + j] = 12
        p["bm"][j] = -6

        p["Wm"][4 + j, j] = 12
        p["Wma"][4 + j, action] = 24
        p["bm"][4 + j] = -24

        p["Wma"][8 + j, action] = 12
        p["bm"][8 + j] = -6

    p["Wmz"][13, 18] = 12
    p["Wma"][13, 6] = -24
    p["bm"][13] = -6

    p["bo"][:] = -1000
    for j, action in enumerate(actions):
        p["bo"][action] = 0
        p["Wo"][action, j] = 1
        p["Wo"][action, 4 + j] = -20
        p["Wo"][action, 128 + 8 + opposite[j]] = -0.03

    # Small geometric tie-breaker.
    epsilon = 0.001
    for action, positive, negative in (
        (1, 20, 22), (2, 22, 20),
        (3, 21, 23), (4, 23, 21),
    ):
        p["Wo"][action, positive] += epsilon
        p["Wo"][action, negative] -= epsilon

    p["bo"][6] = 35.8
    for j in range(4):
        p["Wo"][6, 12 + j] = 1
        p["Wo"][6, 128 + 4 + j] = 8

    p["Wo"][6, 16] = 0.5
    p["Wo"][6, 17] = 0.5
    p["Wo"][6, 128 + 13] = 4

    return p
