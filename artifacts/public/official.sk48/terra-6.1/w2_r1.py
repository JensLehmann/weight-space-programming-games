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
    W3, b3 = p["W3"], p["b3"]

    corners = ((1, 1), (1, 4), (4, 1), (4, 4))
    centre = ((2, 2), (2, 3), (3, 2), (3, 3))

    # Field and toolbar tile detectors.
    for k, colour in enumerate(range(7, 15)):
        for i, j in corners:
            W1[k, colour, i, j] = 1
        W1[k, 0, 2, 2] = -2
        b1[k] = -3

    # Selected head and any head.
    for out in (8, 9):
        for i, j in centre:
            W1[out, 6, i, j] = 1
            W1[out, 15, i, j] = 1
        for i, j in corners:
            W1[out, 0, i, j] = 1
            if out == 9:
                W1[out, 4, i, j] = 1
        b1[out] = -7

    W1[10, 1, 2, 2] = 1
    W1[11, 17, 2, 2] = 1
    W1[12, 18, 2, 2] = 1
    W1[12, 16, 2, 2] = -1

    W1[13, 16, 2, 2] = 63
    b1[13] = -53
    W1[14, 16, 2, 2] = -63
    b1[14] = 54

    # Vertical rail immediately above a tile.
    for colour in (2, 3):
        W1[15, colour, 0, 2] = 1
        W1[15, colour, 0, 3] = 1
    b1[15] = -1

    def w2(out, inp, value):
        W2[out, inp, 1, 1] = value

    for k in range(8):
        w2(k, k, 1)
        w2(k, 13, -1)
        w2(8 + k, k, 1)
        w2(8 + k, 14, -1)

    w2(16, 8, 1)
    w2(16, 13, -1)
    w2(17, 8, 1)
    w2(17, 14, -1)
    w2(18, 9, 1)
    w2(18, 14, -1)
    w2(19, 10, 1)
    w2(19, 13, -1)
    w2(20, 11, 1)
    w2(21, 12, 1)

    for k in range(8):
        w2(22, k, 1)
    w2(22, 15, 1)
    w2(22, 13, -1)
    p["b2"][22] = -1

    # Coordinates of field tiles and indicators for toolbar tiles.
    for k in range(8):
        W3[k, k] = 1
        W3[k, 20] = 1
        b3[k] = -1

        W3[8 + k, k] = 1
        W3[8 + k, 21] = 1
        b3[8 + k] = -1

        W3[16 + k, 8 + k] = 1

    W3[24, 16] = 1
    W3[24, 21] = 1
    b3[24] = -1

    W3[25, 19] = 1
    W3[25, 20] = 1
    b3[25] = -1

    W3[26, 17] = 1
    W3[27, 18] = 1

    for k in range(8):
        W3[28, 8 + k] = 1
    W3[28, 20] = 1
    b3[28] = -1

    W3[29, 16] = 1
    W3[29, 20] = 1
    b3[29] = -1

    W3[30, 17] = 1
    W3[30, 20] = 1
    b3[30] = -1

    W3[31, 22] = 1
    W3[31, 32 + 16] = 1
    b3[31] = -1

    # Propagate from the selected toolbar head to the first
    # unmatched tile.
    p["Wh"][0, 0, 1, 0] = 1
    p["Uh"][0, 26] = 3
    p["Uh"][0, 27] = -2
    for k in range(8):
        p["Uh"][0, 16 + k] = -2
        p["Wh"][1 + k, 0, 1, 0] = 1
        p["Uh"][1 + k, 16 + k] = 1
        p["bh"][1 + k] = -1

    p["Wh"][9, 0, 1, 0] = 1
    p["Uh"][9, 28] = 1
    p["bh"][9] = -1

    # Unselected toolbar head for clicks.
    p["Uh"][10, 27] = 1
    p["Uh"][10, 26] = -1
    p["Wc"][0, 32 + 10] = 1

    # g offsets: h maxima at 0, f3 maxima at 32.
    # For each colour, paired ReLUs encode clipped up, down,
    # right and reached indicators.
    for k in range(8):
        tile_x = 32 + k
        tile_inv_y = 32 + 8 + k
        head_inv_y = 32 + 24
        rail_x = 32 + 25

        for direction in range(4):
            a = 8 * k + 2 * direction
            b = a + 1

            for unit in (a, b):
                if direction == 0:
                    p["Wz"][unit, tile_inv_y] = 63
                    p["Wz"][unit, head_inv_y] = -63
                elif direction == 1:
                    p["Wz"][unit, tile_inv_y] = -63
                    p["Wz"][unit, head_inv_y] = 63
                elif direction == 2:
                    p["Wz"][unit, tile_x] = 63
                    p["Wz"][unit, rail_x] = -63
                    p["bz"][unit] = 2
                else:
                    p["Wz"][unit, tile_x] = -63
                    p["Wz"][unit, rail_x] = 63
                    p["bz"][unit] = -1

            p["bz"][b] -= 1

    for k in range(8):
        p["Wz"][64 + k, 1 + k] = 1

    # Retraction-complete measure.
    def add_completion_expression(unit):
        p["Wz"][unit, 9] += 63
        p["Wz"][unit, 32 + 30] -= 63
        p["Wz"][unit, 32 + 25] -= 63
        p["Wz"][unit, 32 + 29] += 63
        p["bz"][unit] -= 2

    add_completion_expression(72)

    # Colour-specific reload signals when retraction is complete.
    for k in range(8):
        unit = 73 + k
        add_completion_expression(unit)
        p["Wz"][unit, 1 + k] += 100
        p["bz"][unit] -= 100

    p["Wz"][81, 32 + 31] = 1

    # First unmatched target absent; another toolbar head present.
    p["bz"][82] = 1
    for k in range(8):
        p["Wz"][82, 1 + k] = -1
    p["Wz"][83, 10] = 1
    p["Wo"][6, 82] = 500
    p["Wo"][6, 83] = 500
    p["bo"][6] = -750

    # Memory 0..7 holds the target colour through an extension.
    for k in range(8):
        p["Wm"][k, k] = 8
        p["Wmz"][k, 64 + k] = 6
        p["Wmz"][k, 72] = -12
        p["Wmz"][k, 73 + k] = 20
        p["bm"][k] = -3

    # Memory 8 is a constant episode-initialization flag.
    p["bm"][8] = 1

    # Memory 10..17 indicates that the held target has been reached.
    for k in range(8):
        unit = 10 + k
        p["Wm"][unit, k] = 4
        p["bm"][unit] = -6

        up = 8 * k
        down = 8 * k + 2
        reached = 8 * k + 6

        p["Wmz"][unit, up] -= 4
        p["Wmz"][unit, up + 1] += 4
        p["Wmz"][unit, down] -= 4
        p["Wmz"][unit, down + 1] += 4
        p["Wmz"][unit, reached] += 4
        p["Wmz"][unit, reached + 1] -= 4

    # Memory 9 latches the retraction phase.
    p["Wm"][9, 9] = 3
    p["Wm"][9, 8] = 35.0 / np.tanh(1.0)
    for k in range(8):
        p["Wm"][9, 10 + k] = 5
    p["Wmz"][9, 72] = -20
    p["Wma"][9, 6] = -20

    # Memory 18..41 holds gated direction indicators.
    for k in range(8):
        for direction in range(3):
            unit = 18 + 3 * k + direction
            z = 8 * k + 2 * direction
            p["Wm"][unit, k] = 4
            p["Wmz"][unit, z] = 4
            p["Wmz"][unit, z + 1] = -4
            p["bm"][unit] = -6

            action = (1, 2, 4)[direction]
            p["Wo"][action, 128 + unit] += 10

    for action in (1, 2, 4):
        p["bo"][action] += 70

    # Retract immediately on reaching a target, then continue
    # until the arm reaches the desired length.
    p["Wo"][3, 128 + 9] = 150
    for k in range(8):
        p["Wo"][3, 128 + 10 + k] = 100
    p["bo"][3] = 770

    # Move down rather than retract through a crossing rail.
    p["Wm"][42, 9] = 4
    p["Wmz"][42, 81] = 4
    p["bm"][42] = -6
    p["Wo"][2, 128 + 42] += 100
    p["bo"][2] += 100

    # Harmless initialization action while the target memory loads.
    p["bo"][7] = 1

    return p
