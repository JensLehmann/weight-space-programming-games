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

    def detector(out, pixels):
        for dy, dx, color in pixels:
            p["W1"][out, color, dy + 2, dx + 2] += 2.0
        p["b1"][out] = 1.0 - 2.0 * len(pixels)

    # Player centre, for right- and left-facing sprites.
    detector(0, [
        (0, 0, 9), (-1, 0, 9),
        (0, 1, 11), (-1, 1, 11),
    ])
    detector(1, [
        (0, 0, 9), (-1, 0, 9),
        (0, -1, 11), (-1, -1, 11),
    ])

    # Wall and solid block centres.
    detector(2, [
        (0, 0, 5), (-2, -1, 3),
        (1, -2, 3), (2, 1, 3),
    ])
    for out, color in [(3, 14), (4, 8), (5, 12), (8, 15)]:
        detector(out, [
            (0, 0, color),
            (-2, -2, 3), (-2, 2, 3),
            (2, -2, 3), (2, 2, 3),
        ])

    # Spike centres in either orientation.
    for out, dy in [(6, 2), (7, -2)]:
        detector(out, [
            (0, 0, 15), (0, -1, 15), (0, 1, 15),
            (-1, 0, 15), (1, 0, 15),
            (dy, 0, 0),
        ])

    # Goal cross.
    detector(9, [
        (0, 0, 7), (0, -1, 7), (0, 1, 7),
        (-1, 0, 7), (1, 0, 7),
    ])
    p["W1"][10, 16, 2, 2] = 1.0

    # Detectors displaced two pixels vertically. W2 and the
    # recurrent block extend these displacements to one tile.
    shifted = {}
    index = 11
    for direction in (1, -1):
        shifted[direction] = []

        if direction == 1:
            wall = [
                (-1, -2, 3), (0, 1, 3),
                (0, 0, 5), (-1, 0, 5),
            ]
        else:
            wall = [
                (0, -1, 3), (0, 0, 5),
                (1, 0, 5), (2, 0, 5),
            ]
        detector(index, wall)
        shifted[direction].append(index)
        index += 1

        for color in (14, 8, 12, 15):
            if direction == 1:
                pixels = [
                    (-2, 0, color), (-1, 0, color), (0, 0, color),
                    (0, -2, 3), (0, 2, 3),
                ]
            else:
                pixels = [
                    (0, 0, color), (1, 0, color), (2, 0, color),
                    (0, -2, 3), (0, 2, 3),
                ]
            detector(index, pixels)
            shifted[direction].append(index)
            index += 1

        if direction == 1:
            spike = [
                (-2, -1, 15), (-2, 0, 15),
                (-2, 1, 15), (-1, 0, 15),
            ]
        else:
            spike = [
                (2, -1, 15), (2, 0, 15),
                (2, 1, 15), (1, 0, 15),
            ]
        detector(index, spike)
        shifted[direction].append(index)
        index += 1

    # f2: tile-centred detections.
    p["W2"][0, 0, 1, 1] = 1.0
    p["W2"][0, 1, 1, 1] = 1.0
    for dst, src in [(1, 2), (2, 3), (3, 4), (4, 5), (6, 8)]:
        p["W2"][dst, src, 1, 1] = 1.0
    p["W2"][5, 6, 1, 1] = 1.0
    p["W2"][5, 7, 1, 1] = 1.0
    p["W2"][7, 9, 0, 1] = 1.0
    p["W2"][8, 10, 1, 1] = 1.0

    # Camera-relative gravity indicators at the player.
    p["W2"][9, 0, 1, 1] = 1.0
    p["W2"][9, 1, 1, 1] = 1.0
    p["W2"][9, 10, 1, 1] = 1.0
    p["b2"][9] = -1.5

    p["W2"][10, 0, 1, 1] = 1.0
    p["W2"][10, 1, 1, 1] = 1.0
    p["W2"][10, 10, 1, 1] = -1.0
    p["b2"][10] = -0.5

    for src in shifted[1]:
        p["W2"][11, src, 0, 1] = 1.0
    for src in shifted[-1]:
        p["W2"][12, src, 2, 1] = 1.0

    # f3: player, player row, tile columns, blocking and clickable
    # tiles, spikes, ray-stopping tiles, and goal.
    p["W3"][0, 0] = 1.0
    p["W3"][1, 32 + 0] = 1.0
    for src in range(8):
        p["W3"][2, 64 + src] = 1.0
        p["W3"][3, 64 + src] = 1.0
    p["b3"][3] = -1.0

    for src in (1, 2, 3, 4, 5, 6):
        p["W3"][4, src] = 1.0
    for src in (2, 3, 4, 6):
        p["W3"][5, src] = 1.0
    p["W3"][6, 5] = 1.0
    for src in (1, 2, 3, 4, 6, 7):
        p["W3"][7, src] = 1.0
    p["W3"][8, 7] = 1.0
    p["W3"][9, 11] = 1.0
    p["W3"][10, 12] = 1.0

    # Row-broadcast gravity indicators.
    for out in (11, 12):
        p["W3"][out, 32 + 9] = 8.4
    p["b3"][12] = -1.0
    for out in (13, 14):
        p["W3"][out, 32 + 10] = 25.2
    p["b3"][14] = -1.0

    # Triangular gates for clickable tiles one row above or below
    # the player, using row and column maxima.
    click_sources = (2, 3, 4, 6)
    for j, bias in enumerate((-232.0, -233.0, -234.0)):
        out = 15 + j
        for src in click_sources:
            p["W3"][out, src] = 100.0
        p["W3"][out, 64 + 9] = 840.0
        p["W3"][out, 8] = 63.0
        p["b3"][out] = bias

    for j, bias in enumerate((-234.0, -235.0, -236.0)):
        out = 18 + j
        for src in click_sources:
            p["W3"][out, src] = 100.0
        p["W3"][out, 64 + 10] = 2520.0
        p["W3"][out, 8] = 63.0
        p["b3"][out] = bias

    p["W3"][21, 3] = 1.0

    # Blocking maps shifted by one grid cell in each direction.
    p["Uh"][0, 9] = 1.0
    p["Wh"][1, 0, 0, 1] = 1.0
    p["Wh"][2, 1, 0, 1] = 1.0

    p["Uh"][3, 10] = 1.0
    p["Wh"][4, 3, 2, 1] = 1.0
    p["Wh"][5, 4, 2, 1] = 1.0

    # Spike rays, stopped by other solid tiles.
    p["Wh"][6, 6, 0, 1] = 1.0
    p["Wh"][7, 7, 2, 1] = 1.0
    for out in (6, 7):
        p["Uh"][out, 6] = 100.0
        p["Uh"][out, 7] = -10000.0
        p["bh"][out] = -0.5

    # Horizontal reachability along the player's row.
    for out in (8, 9):
        p["Uh"][out, 0] = 66.0
        p["Uh"][out, 1] = 10000.0
        p["Uh"][out, 4] = -10000.0
        p["bh"][out] = -10001.0
    p["Wh"][8, 8, 1, 0] = 1.0
    p["Wh"][9, 9, 1, 2] = 1.0

    # Candidate movement destinations.
    for out, reach, is_up in [
        (10, 8, True), (11, 9, True),
        (12, 8, False), (13, 9, False),
    ]:
        p["Wh"][out, reach, 1, 1] = 1.0
        if is_up:
            p["Wh"][out, 2, 0, 1] = -10000.0
            p["Wh"][out, 6, 1, 1] = -2.0
            p["Uh"][out, 11] = 10000.0
            p["Uh"][out, 12] = -10000.0
        else:
            p["Wh"][out, 5, 2, 1] = -10000.0
            p["Wh"][out, 7, 1, 1] = -2.0
            p["Uh"][out, 13] = 10000.0
            p["Uh"][out, 14] = -10000.0

        p["Uh"][out, 0] = -10000.0
        p["Uh"][out, 1] = 10000.0
        p["Uh"][out, 2] = 10000.0
        p["Uh"][out, 3] = -10000.0
        p["Uh"][out, 4] = -10000.0
        p["Uh"][out, 8] = 40.0
        p["bh"][out] = -30000.0

    # Diffuse reachable-column strength vertically. The coefficient
    # sum is less than one, keeping the tied recurrence bounded.
    p["Wh"][15, 8, 1, 1] = 1.0
    p["Wh"][15, 9, 1, 1] = 1.0
    p["Wh"][15, 15, 0, 1] = 0.48
    p["Wh"][15, 15, 2, 1] = 0.48

    # Click score for reachable columns and adjacent blocks.
    p["Wh"][14, 15, 1, 1] = 0.2
    p["Wh"][14, 8, 1, 0] = 0.6
    p["Wh"][14, 9, 1, 2] = 0.6
    p["Wh"][14, 6, 0, 1] = -4.0

    for j, weight in enumerate((90.0, -180.0, 90.0)):
        p["Uh"][14, 15 + j] = weight
    for j, weight in enumerate((500.0, -1000.0, 500.0)):
        p["Uh"][14, 18 + j] = weight

    p["Uh"][14, 5] = 10000.0
    p["Uh"][14, 21] = 2.0
    p["bh"][14] = -10000.0

    # Action and click-map readouts.
    p["Wz"][0, 10] = 1.0
    p["Wz"][0, 12] = 1.0
    p["Wz"][1, 11] = 1.0
    p["Wz"][1, 13] = 1.0
    p["Wz"][2, 14] = 1.0

    p["bo"][:] = -10000.0
    p["bo"][3] = 0.0
    p["bo"][4] = 0.0
    p["bo"][6] = 0.0
    p["Wo"][3, 1] = 1.0
    p["Wo"][4, 0] = 1.0
    p["Wo"][6, 2] = 1.0

    p["Wc"][0, 32 + 14] = 1.0
    p["Wc"][0, 5] = 0.0001

    return p
