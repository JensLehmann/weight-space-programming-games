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

    def detector(channel, pixels):
        for dy, dx, colors in pixels:
            if isinstance(colors, int):
                colors = (colors,)
            for color in colors:
                p["W1"][channel, color, dy + 2, dx + 2] += 1.0
        p["b1"][channel] = 1.0 - len(pixels)

    peg_shape = (
        [(-1, 0), (-1, 1)]
        + [(0, x) for x in (-1, 0, 1, 2)]
        + [(1, x) for x in (-1, 0, 1, 2)]
        + [(2, 0), (2, 1)]
    )
    for channel, color in ((0, 14), (1, 8), (2, 9)):
        detector(channel, [(dy, dx, color) for dy, dx in peg_shape])

    detector(3, [
        (-1, -1, 15), (-1, 0, 0), (-1, 1, 7), (-1, 2, 15),
        (0, -1, 15), (0, 0, 7), (0, 1, 7), (0, 2, 15),
        (1, 0, 15), (1, 1, 15),
    ])

    interior = [
        (dy, dx)
        for dy in (-1, 0, 1, 2)
        for dx in (-1, 0, 1, 2)
    ]
    detector(4, [(dy, dx, 1) for dy, dx in interior])
    detector(5, [(dy, dx, 12) for dy, dx in interior])

    ring = [
        (-1, 0), (-1, 1),
        (0, -1), (0, 2),
        (1, -1), (1, 2),
        (2, 0), (2, 1),
    ]
    detector(6, [(dy, dx, 2) for dy, dx in ring])

    detector(7, [
        (-2, -1, 3), (-2, 0, 3), (-2, 1, 3), (-2, 2, 3),
        (0, -2, 3), (1, -2, 3),
    ])

    detector(8, [
        (0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0),
        (-1, -1, 5), (-1, 0, 5), (-1, 1, 5), (-1, 2, 5),
        (0, -1, 5), (0, 2, 5), (1, -1, 5), (1, 2, 5),
    ])

    platform_border = (
        [(-2, dx, 11) for dx in (-2, -1, 0, 1, 2)]
        + [(dy, -2, 11) for dy in (-1, 0, 1, 2)]
    )
    detector(9, platform_border)

    # Immediate rail beyond an orange platform edge. The second
    # detector for each direction also requires a peg on the platform.
    edge_patterns = [
        ([(0, 0, 11), (0, 1, 11), (-1, 0, 5), (-1, 1, 5)],
         (2, 0)),
        ([(0, 0, 11), (0, 1, 11), (1, 0, 5), (1, 1, 5)],
         (-2, 0)),
        ([(0, 0, 11), (1, 0, 11), (0, -1, 5), (1, -1, 5)],
         (0, 2)),
        ([(0, 0, 11), (1, 0, 11), (0, 1, 5), (1, 1, 5)],
         (0, -2)),
    ]
    for i, (edge, peg_offset) in enumerate(edge_patterns):
        detector(10 + i, edge)
        dy, dx = peg_offset
        detector(14 + i, edge + [(dy, dx, (14, 8, 9))])

    for i in range(18):
        p["W2"][i, i, 1, 1] = 1.0
    for i in range(9):
        p["W3"][i, i] = 1.0

    # Midpoint, landing and candidate-click indicators.
    p["W3"][9, 0] = 1.0
    p["W3"][9, 1] = 1.0
    p["W3"][9, 2] = 0.25
    p["W3"][9, 3] = 0.25

    p["W3"][10, 4] = 1.0
    p["W3"][10, 5] = 1.0
    p["W3"][10, 7] = 1.0

    for i in (0, 1, 2, 6):
        p["W3"][11, i] = 1.0
    p["W3"][12, 9] = 1.0

    for i in range(4):
        p["W3"][13 + i, 10 + i] = 1.0
        p["W3"][17 + i, 14 + i] = 1.0

    # Directional waves detect midpoint and landing cells six and
    # twelve pixels away.
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for i, (dy, dx) in enumerate(directions):
        mid, target, excess, score = i, 4 + i, 8 + i, 12 + i

        p["Uh"][mid, 9] = 1.0
        p["Wh"][mid, mid, 1 + dy, 1 + dx] = 0.5

        p["Uh"][target, 10] = 1.0
        p["Wh"][target, target, 1 + dy, 1 + dx] = 0.5

        p["Wh"][excess, target, 1, 1] = 4096.0
        p["Uh"][excess, 10] = -4096.0
        p["bh"][excess] = -1.0

        p["Wh"][score, mid, 1, 1] = 64.0
        p["Wh"][score, target, 1, 1] = 4096.0
        p["Wh"][score, excess, 1, 1] = -1.0
        p["Uh"][score, 9] = -64.0
        p["Uh"][score, 10] = -4096.0
        p["Uh"][score, 11] = 100.0
        p["bh"][score] = -101.15

    for i in range(12, 16):
        p["Wz"][0, i] = 1.0
    p["bz"][0] = -0.045
    p["Wz"][1, 32 + 6] = 1.0
    p["Wz"][2, 32 + 8] = 1.0

    for i in range(4):
        p["Wz"][3 + i, 32 + 13 + i] = 1.0
        p["Wz"][7 + i, 32 + 17 + i] = 1.0

    # Visible peg count from the spatial means.
    for i in (0, 1, 2):
        p["Wz"][32, 64 + i] = 4096.0

    # Fallback exploration clock.
    p["Wm"][0, 15] = -5.0
    p["bm"][0] = 0.25
    for i in range(1, 16):
        p["Wm"][i, i - 1] = 5.0

    # Previous peg count, marker presence, and initialized flag.
    p["Wmz"][16, 32] = 0.005
    p["Wmz"][18, 1] = 4.0
    p["bm"][28] = 5.0

    # A click after a displayed marker is treated as a non-removing
    # jump if the visible peg count did not decrease.
    p["Wm"][17, 16] = -1200.0
    p["Wmz"][17, 32] = 6.0
    p["Wm"][17, 18] = 50.0
    p["Wma"][17, 6] = 50.0
    p["Wm"][17, 28] = 50.0
    p["bm"][17] = -146.0

    # Short exploration interval following such a jump.
    p["Wm"][19, 17] = 4.0
    p["bm"][19] = -2.0
    for i in range(20, 27):
        p["Wm"][i, i - 1] = 4.0
        p["bm"][i] = -2.0

    # Remember the immediately preceding direction for momentum.
    for i in range(4):
        p["Wma"][29 + i, 1 + i] = 4.0

    m0, m8 = 128, 136
    p["Wo"][1, m0] = 0.3
    p["Wo"][1, m8] = -0.3
    p["Wo"][2, m0] = -0.3
    p["Wo"][2, m8] = 0.3
    p["Wo"][3, m0] = -0.3
    p["Wo"][3, m8] = -0.3
    p["Wo"][4, m0] = 0.3
    p["Wo"][4, m8] = 0.3
    p["bo"][4] = 0.02

    opposite = (1, 0, 3, 2)
    for i in range(4):
        action = 1 + i
        p["Wo"][action, 3 + i] = 4.0
        p["Wo"][action, 7 + i] = 5.0
        p["Wo"][action, 128 + 29 + i] = 0.8
        p["Wo"][action, 128 + 29 + opposite[i]] = -0.8

    p["Wo"][6, 0] = 800.0
    p["Wo"][6, 1] = 200.0
    p["Wo"][6, 2] = 200.0
    p["bo"][6] = -0.4

    for i in [17] + list(range(19, 27)):
        p["Wo"][6, 128 + i] = -100.0
        p["bo"][6] -= 100.0

    p["Wc"][0, 6] = 10.0
    p["Wc"][0, 8] = 20.0
    p["Wc"][0, 12] = 0.01
    for i in range(12, 16):
        p["Wc"][0, 32 + i] = 1.0

    return p
