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

    # Player detection anchored above the player.
    p["W1"][0, 9, 3, 2] = 1
    p["W1"][0, 9, 4, 2] = 1
    p["W1"][0, 11, 4, 1] = 1
    p["W1"][0, 11, 4, 3] = 1
    p["b1"][0] = -2

    # Open background and goal.
    p["W1"][1, 10, 2, 2] = 1
    p["W1"][1, 7, 2, 2] = 1

    # Centers of full colored blocks.
    for channel, color in ((2, 14), (3, 12), (4, 15), (16, 8)):
        for y in range(5):
            for x in range(5):
                if y in (0, 4) and x in (0, 4):
                    p["W1"][channel, color, y, x] = -5
                else:
                    p["W1"][channel, color, y, x] = 1
        p["b1"][channel] = -20

    # Player detection anchored at the center of the player's tile.
    p["W1"][5, 9, 0, 2] = 1
    p["W1"][5, 9, 1, 2] = 1
    for y in (1, 2):
        p["W1"][5, 11, y, 1] = 1
        p["W1"][5, 11, y, 3] = 1
    p["b1"][5] = -3

    p["W1"][6, 7, 2, 2] = 1

    # Spike stripe at three horizontal offsets.
    for channel, dx in ((7, -1), (8, 0), (9, 1)):
        p["W1"][channel, 0, 2, 2 + dx] = 1
        p["W1"][channel, 11, 2, 1 + dx] = 1
        p["W1"][channel, 11, 2, 3 + dx] = 1
        p["b1"][channel] = -2

    # Two thresholds form a clipped indicator for a solid gray region.
    for channel, threshold in ((10, 19), (11, 20)):
        p["W1"][channel, 5, :, :] = 1
        p["b1"][channel] = -threshold

    for channel, color in ((12, 14), (13, 12), (14, 15), (15, 8)):
        p["W1"][channel, color, 2, 2] = 1

    # Player source one pixel above the upper-anchored detector.
    p["W2"][2, 0, 2, 1] = 1
    p["W2"][3, 1, 1, 1] = 1

    for channel in (2, 3, 4):
        p["W2"][4, channel, 1, 1] = 1

    for channel in (7, 8, 9):
        for x in range(3):
            p["W2"][5, channel, 1, x] = 1

    p["W2"][6, 10, 1, 1] = 1
    p["W2"][6, 11, 1, 1] = -1

    for output, source in ((7, 12), (8, 13), (9, 14), (10, 15)):
        p["W2"][output, source, 1, 1] = 1

    p["W2"][11, 16, 1, 1] = 1
    p["W2"][12, 5, 1, 1] = 1
    p["W2"][13, 6, 1, 1] = 1

    for channel in range(14):
        p["W3"][channel, channel] = 1

    # Downward spike-risk ray.
    p["Wh"][0, 0, 0, 1] = 0.99
    p["Uh"][0, 5] = 1
    p["Uh"][0, 6] = -2
    for feature in (7, 8, 9, 10):
        p["Uh"][0, feature] = -0.35

    # Same-row rays from the player.
    p["Uh"][1, 12] = 1
    p["Wh"][1, 1, 1, 2] = 0.96

    p["Uh"][5, 12] = 1
    p["Wh"][5, 5, 1, 0] = 0.96

    # Same-row breakable blocks.
    p["Wh"][2, 1, 1, 1] = 1
    p["Uh"][2, 4] = 2
    p["bh"][2] = -2

    p["Wh"][6, 5, 1, 1] = 1
    p["Uh"][6, 4] = 2
    p["bh"][6] = -2

    # Seed rays on the row one tile above the player.
    p["Uh"][3, 2] = 1
    p["Wh"][7, 3, 2, 1] = 1

    p["Wh"][4, 7, 2, 2] = 0.96
    p["Wh"][4, 4, 1, 2] = 0.96

    p["Wh"][8, 7, 2, 0] = 0.96
    p["Wh"][8, 8, 1, 0] = 0.96

    # Overhead openings and blocks, with spike-risk suppression.
    for output, ray, feature in (
        (9, 4, 3),
        (10, 8, 3),
        (11, 4, 4),
        (12, 8, 4),
    ):
        p["Wh"][output, ray, 1, 1] = 1
        p["Wh"][output, 0, 1, 1] = -10
        p["Uh"][output, feature] = 2
        p["bh"][output] = -2

    # Same-row goal signals.
    p["Wh"][13, 1, 1, 1] = 1
    p["Uh"][13, 13] = 2
    p["bh"][13] = -2

    p["Wh"][14, 5, 1, 1] = 1
    p["Uh"][14, 13] = 2
    p["bh"][14] = -2

    # Breakable block immediately above the player.
    p["Wh"][15, 7, 2, 1] = 1
    p["Wh"][15, 0, 1, 1] = -10
    p["Uh"][15, 4] = 2
    p["bh"][15] = -2

    for channel in range(16):
        p["Wz"][channel, channel] = 1

    # max(opening, discounted block) for each direction.
    p["Wz"][20, 9] = 1
    p["Wz"][20, 11] = -0.7
    p["Wz"][21, 11] = 0.7

    p["Wz"][22, 10] = 1
    p["Wz"][22, 12] = -0.7
    p["Wz"][23, 12] = 0.7

    # Green switch fallback when no overhead candidate is present.
    p["Wz"][24, 32 + 11] = 1
    for channel in (9, 10, 11, 12, 15):
        p["Wz"][24, channel] = -4
    p["bz"][24] = -0.5

    # Same-row block click requests, biased toward the side with the
    # better overhead candidate.
    p["Wz"][25, 2] = 1
    p["Wz"][25, 9] = 2
    p["Wz"][25, 11] = 1.4
    p["Wz"][25, 10] = -2
    p["Wz"][25, 12] = -1.4
    p["bz"][25] = -0.2

    p["Wz"][26, 6] = 1
    p["Wz"][26, 10] = 2
    p["Wz"][26, 12] = 1.4
    p["Wz"][26, 9] = -2
    p["Wz"][26, 11] = -1.4
    p["bz"][26] = -0.2

    # Retain a weak signed movement preference through clicks.
    p["Wm"][0, 0] = 1
    p["Wma"][0, 3] = -2
    p["Wma"][0, 4] = 2

    p["Wo"][3, 20] = 1
    p["Wo"][3, 21] = 1
    p["Wo"][3, 13] = 3
    p["Wo"][3, 128] = -0.03

    p["Wo"][4, 22] = 1
    p["Wo"][4, 23] = 1
    p["Wo"][4, 14] = 3
    p["Wo"][4, 128] = 0.03
    p["bo"][4] = 0.001

    p["Wo"][6, 15] = 1.2
    p["Wo"][6, 24] = 2.2
    p["Wo"][6, 25] = 1.5
    p["Wo"][6, 26] = 1.5

    # Click the overhead block, a same-row block, or the green switch.
    p["Wc"][0, 32 + 15] = 2
    p["Wc"][0, 32 + 2] = 1
    p["Wc"][0, 32 + 6] = 1
    p["Wc"][0, 11] = 0.3

    return p
