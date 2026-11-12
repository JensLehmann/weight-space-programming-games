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

    def pattern(channel, color, required, forbidden=()):
        for dy, dx in required:
            p["W1"][channel, color, dy + 2, dx + 2] += 2.0
        for dy, dx in forbidden:
            p["W1"][channel, color, dy + 2, dx + 2] -= 2.0
        p["b1"][channel] = 1.0 - 2.0 * len(required)

    # All tile detectors use the tile's (2, 2) pixel as their anchor.
    disc = [
        (-1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 1), (0, 2),
        (1, -1), (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1),
    ]
    corners = [(-1, -1), (-1, 2), (2, -1), (2, 2)]
    pattern(0, 14, disc, corners)  # green piece
    pattern(1, 8, disc, corners)   # red piece
    pattern(2, 9, disc, corners)   # blue piece

    purple = (
        [(-2, x) for x in range(-1, 3)]
        + [(-1, -1), (-1, 2), (0, -1), (0, 2)]
        + [(1, x) for x in range(-1, 3)]
    )
    pattern(3, 15, purple, [(-1, 0), (-1, 1), (0, 0), (0, 1)])

    interior = [(y, x) for y in range(-1, 3) for x in range(-1, 3)]
    pattern(4, 1, interior)
    pattern(5, 12, interior)

    ring = [
        (-1, 0), (-1, 1),
        (0, -1), (0, 2),
        (1, -1), (1, 2),
        (2, 0), (2, 1),
    ]
    pattern(6, 2, ring, [(0, 0), (0, 1), (1, 0), (1, 1)])
    pattern(7, 2, disc, corners)
    pattern(8, 3, [(-2, 0), (0, -2)])

    p["W1"][10, 16, 2, 2] = 1.0
    p["W1"][11, 17, 2, 2] = 1.0

    # Four platform corners, with the final pixel of each side supplied
    # by a one-pixel offset in the second convolution.
    for ch, dy, dx in (
        (12, -2, -2), (13, -2, 2),
        (14, 2, -2), (15, 2, 2),
    ):
        p["W1"][ch, 11, dy + 2, dx + 2] = 1.0

    for i in list(range(8)) + [10, 11]:
        p["W2"][i, i, 1, 1] = 1.0

    # Selected piece.
    p["W2"][8, 8, 1, 1] = 1.0
    for i in range(3):
        p["W2"][8, i, 1, 1] = 1.0
    p["b2"][8] = -1.0

    # Yellow-corner platform detector.
    p["W2"][12, 12, 1, 1] = 1.0
    p["W2"][12, 13, 1, 2] = 1.0
    p["W2"][12, 14, 2, 1] = 1.0
    p["W2"][12, 15, 2, 2] = 1.0
    p["b2"][12] = -3.0

    for i in range(4):
        p["W3"][i, i] = 1.0

    # Empty or selected endpoint, ring, midpoint, movable piece, failure.
    for i in (4, 5, 6, 8):
        p["W3"][4, i] = 1.0
    p["W3"][5, 6] = 1.0
    p["W3"][6, 0:4] = 1.0
    p["W3"][7, 0:3] = 1.0
    p["W3"][8, 7] = 1.0
    p["W3"][9, 8] = 1.0

    # Position-based click tie-breakers.
    p["W3"][10, 0:3] = 1.0
    p["W3"][10, 10] = -0.3
    p["W3"][10, 11] = -0.3

    p["W3"][11, 6] = 1.0
    p["W3"][11, 10] = 0.3
    p["W3"][11, 11] = 0.3
    p["b3"][11] = -0.6

    # Platform presence and its extrema.
    p["W3"][12, 12] = 1.0
    for out, coord, sign in (
        (13, 10, 1.0), (14, 10, -1.0),
        (15, 11, 1.0), (16, 11, -1.0),
    ):
        p["W3"][out, 12] = 1.0
        p["W3"][out, coord] = sign
        if sign > 0:
            p["b3"][out] = -1.0

    # Movable-piece extrema.
    for out, coord, sign in (
        (17, 10, 1.0), (18, 10, -1.0),
        (19, 11, 1.0), (20, 11, -1.0),
    ):
        p["W3"][out, 0:3] = 1.0
        p["W3"][out, coord] = sign
        if sign > 0:
            p["b3"][out] = -1.0

    # Piece on a platform, plus its position.
    p["W3"][21, 12] = 1.0
    p["W3"][21, 0:3] = 1.0
    p["b3"][21] = -1.0
    for out, coord, sign in (
        (22, 10, 1.0), (23, 10, -1.0),
        (24, 11, 1.0), (25, 11, -1.0),
    ):
        p["W3"][out, 12] = 1.0
        p["W3"][out, 0:3] = 1.0
        p["W3"][out, coord] = sign
        p["b3"][out] = -2.0 if sign > 0 else -1.0

    # Directional jump pipelines.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for d, (dy, dx) in enumerate(directions):
        a, b, c, valid = (4 * d + i for i in range(4))
        ky, kx = 1 + dy, 1 + dx

        p["Wh"][a, a, ky, kx] = 0.25
        p["Uh"][a, 4] = 1.0

        p["Wh"][b, a, ky, kx] = 1024.0
        p["Uh"][b, 6] = 2048.0
        p["Uh"][b, 0] = 0.35
        p["Uh"][b, 1] = 0.30
        p["Uh"][b, 2] = 0.02
        p["Uh"][b, 9] = -512.0
        p["bh"][b] = -2048.5

        p["Wh"][c, c, ky, kx] = 0.25
        p["Wh"][c, b, 1, 1] = 1.0

        p["Wh"][valid, c, ky, kx] = 1024.0
        p["Uh"][valid, 7] = 2048.0
        p["Uh"][valid, 5] = 2048.0
        p["bh"][valid] = -2048.3

        p["Wc"][0, 32 + valid] = 1.0

    p["Wc"][0, 5] = 10.0
    p["Wc"][0, 10] = 0.2
    p["Wc"][0, 11] = 0.2

    # Global features: ring, jump scores, failure, weak jump.
    p["Wz"][0, 32 + 5] = 1.0
    for d in range(4):
        p["Wz"][1, 4 * d + 3] = 1.0
    p["Wz"][2, 32 + 8] = 1.0
    p["Wz"][3, 32 + 5] = 1.0
    for d in range(4):
        p["Wz"][3, 4 * d + 3] = -1.4

    # When no piece is riding a platform, bias toward the upper-left
    # piece extrema. Presence gates suppress these features otherwise.
    approach = [
        (18, 14),  # up
        (14, 18),  # down
        (20, 16),  # left
        (16, 20),  # right
    ]
    for i, (positive, negative) in enumerate(approach):
        z = 4 + i
        p["Wz"][z, 32 + positive] = 1.0
        p["Wz"][z, 32 + negative] = -1.0
        p["Wz"][z, 32 + 12] = 1.0
        p["Wz"][z, 32 + 21] = -2.0
        p["bz"][z] = -1.0

    # A carried piece biases movement toward the opposite extrema.
    chase = [
        (22, 17),  # up
        (17, 22),  # down
        (24, 19),  # left
        (19, 24),  # right
    ]
    for i, (positive, negative) in enumerate(chase):
        z = 8 + i
        p["Wz"][z, 32 + positive] = 1.0
        p["Wz"][z, 32 + negative] = -1.0
        p["Wz"][z, 32 + 21] = 1.0
        p["bz"][z] = -1.0

    # Two oscillators vary exploration.
    p["Wm"][0, 0] = 1.2
    p["Wm"][0, 1] = -0.43
    p["Wm"][1, 0] = 0.43
    p["Wm"][1, 1] = 1.2
    p["bm"][0] = 0.02

    p["Wm"][4, 4] = 1.12
    p["Wm"][4, 5] = -0.19
    p["Wm"][5, 4] = 0.19
    p["Wm"][5, 5] = 1.12
    p["bm"][4] = 0.013
    p["bm"][5] = 0.007

    # A weak jump sets a temporary exploration trace.
    p["Wmz"][2, 3] = 3.0
    p["Wm"][3, 3] = 0.92
    p["Wm"][3, 2] = 3.0

    p["Wo"][4, 128] = 1.0
    p["Wo"][3, 128] = -1.0
    p["Wo"][2, 129] = 1.0
    p["Wo"][1, 129] = -1.0

    p["Wo"][4, 132] = 0.65
    p["Wo"][3, 132] = -0.65
    p["Wo"][2, 133] = 0.65
    p["Wo"][1, 133] = -0.65

    for i, action in enumerate((1, 2, 3, 4)):
        p["Wo"][action, 4 + i] = 2.5
        p["Wo"][action, 8 + i] = 2.5

    p["Wo"][6, 0] = 10.0
    p["Wo"][6, 1] = 12.0
    p["Wo"][6, 131] = -7.0
    p["bo"][6] = -1.0

    p["Wo"][0, 2] = 10.0
    p["bo"][0] = -5.0
    p["bo"][5] = -10.0
    p["bo"][7] = -10.0

    return p
