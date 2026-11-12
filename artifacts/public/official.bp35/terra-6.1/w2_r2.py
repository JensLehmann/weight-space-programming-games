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
    p = {k: np.zeros(shape, dtype=np.float32)
         for k, shape in shapes.items()}

    # Player centre.
    for dy in (-2, 0, 1):
        p["W1"][0, 9, dy + 2, 2] = 1
    p["b1"][0] = -2

    # Unique marker two pixels above the player centre.
    for dy in (0, 1, 2):
        p["W1"][1, 9, dy + 2, 2] = 1
    p["W1"][1, 5, 2, 1] = 1
    p["W1"][1, 5, 2, 3] = 1
    p["b1"][1] = -4

    # Solid block centres.
    for out, colour in ((2, 14), (3, 8), (4, 12), (5, 15)):
        p["W1"][out, colour, 2, 2] = 1
        for dy in (-2, 2):
            for dx in (-2, 2):
                p["W1"][out, 3, dy + 2, dx + 2] = 1
        p["b1"][out] = -4

    # Spikes in the two orientations.
    for out, dy in ((6, 2), (7, -2)):
        for dx in (-1, 0, 1):
            p["W1"][out, 15, 2, dx + 2] = 1
        for dx in (-2, 2):
            p["W1"][out, 5, 2, dx + 2] = 1
        p["W1"][out, 0, dy + 2, 2] = 1
        p["b1"][out] = -5

    p["W1"][8, 7, 2, 2] = 1
    p["W1"][9, 10, 2, 2] = 1

    for colour in (5, 8, 12, 14, 15):
        p["W1"][10, colour, 2, 2] = 1

    # Suppress detections in the status bar.
    p["W1"][11, 16, 2, 2] = 1
    p["b1"][11] = -58.0 / 63.0

    # Two ramps per tile-column centre. W2 combines each pair into
    # a one-pixel triangular peak at columns 3, 9, ..., 57.
    for i in range(10):
        centre = 3 + 6 * i
        a, b = 12 + 2 * i, 13 + 2 * i
        p["W1"][a, 17, 2, 2] = 1
        p["b1"][a] = -(centre - 1) / 63.0
        p["W1"][b, 17, 2, 2] = 1
        p["b1"][b] = -centre / 63.0

    # Local features.
    p["W2"][0, 0, 1, 1] = 1
    p["W2"][1, 1, 2, 1] = 1
    for out, src in ((2, 2), (3, 3), (4, 4), (5, 5),
                     (7, 8), (8, 9), (9, 10), (11, 11)):
        p["W2"][out, src, 1, 1] = 1
    p["W2"][6, 6, 1, 1] = 1
    p["W2"][6, 7, 1, 1] = 1

    # Obstacle plane displaced one pixel upward.
    p["W2"][10, 10, 2, 1] = 1

    for i in range(10):
        p["W2"][12 + i, 12 + 2 * i, 1, 1] = 63
        p["W2"][12 + i, 13 + 2 * i, 1, 1] = -126

    # F3:
    # 0 player; 1 displaced player row; 2 player column;
    # 3 target; 4 spike; 5 barrier; 6 shifted obstacle;
    # 7 clickable; 8 tile-column lattice; 9 green block.
    p["W3"][0, 0] = 1
    p["W3"][1, 32 + 1] = 1
    p["W3"][2, 64 + 0] = 1

    for src in (2, 4, 5, 7, 8):
        p["W3"][3, src] = 1

    p["W3"][4, 6] = 1
    p["W3"][4, 11] = -100

    p["W3"][5, 9] = 1
    p["W3"][5, 6] = -1
    p["W3"][6, 10] = 1

    for src in (2, 4, 5):
        p["W3"][7, src] = 1

    for i in range(10):
        p["W3"][8, 12 + i] = 1
    p["W3"][9, 3] = 1

    # Move the obstacle plane another five pixels upward.
    p["Uh"][0, 6] = 1
    for out in range(1, 6):
        p["Wh"][out, out - 1, 2, 1] = 1

    # Propagate spike exposure downward, stopping at barriers.
    p["Uh"][6, 4] = 1
    p["Uh"][6, 5] = -3
    p["Wh"][6, 6, 0, 1] = 0.995

    # Complete the six-pixel upward shift of the player row.
    p["Wh"][7, 7, 1, 1] = 0
    p["Uh"][7, 1] = 1
    p["Wh"][7, 7, 1, 1] = 0
    p["Wh"][8, 7, 2, 1] = 1
    p["Wh"][9, 8, 2, 1] = 1

    # Safe targets on the lattice, one tile above the player.
    p["Wh"][10, 9, 1, 1] = 1
    p["Uh"][10, 8] = 1
    p["Uh"][10, 3] = 1
    p["Wh"][10, 6, 0, 1] = -4
    p["bh"][10] = -2

    # Directional fields. The shifted obstacle plane blocks propagation
    # through cells that cannot be entered.
    p["Wh"][11, 10, 1, 1] = 1
    p["Wh"][11, 11, 1, 2] = 0.92
    p["Wh"][11, 5, 1, 1] = -4

    p["Wh"][12, 10, 1, 1] = 1
    p["Wh"][12, 12, 1, 0] = 0.92
    p["Wh"][12, 5, 1, 1] = -4

    # Read the directional fields at the player's column, excluding
    # a target directly in that column.
    p["Wh"][13, 11, 1, 2] = 1
    p["Uh"][13, 2] = 100
    p["bh"][13] = -100

    p["Wh"][14, 12, 1, 0] = 1
    p["Uh"][14, 2] = 100
    p["bh"][14] = -100

    # Directly clickable safe target.
    p["Wh"][15, 10, 1, 1] = 1
    p["Uh"][15, 2] = 1
    p["Uh"][15, 7] = 1
    p["bh"][15] = -2

    # Global features: right, left, direct click, green present.
    p["Wz"][0, 13] = 1
    p["Wz"][1, 14] = 1
    p["Wz"][2, 15] = 1
    p["Wz"][3, 32 + 9] = 1

    # A small rightward default preference.
    p["bm"][0] = 0.1

    # Delay before considering a visible green block. A click restarts
    # this delay through the previous-action input.
    p["Wm"][1, 1] = 1
    p["bm"][1] = 0.1
    p["Wma"][1, 6] = -1

    p["bo"][:] = -10000
    p["bo"][3] = 0
    p["bo"][4] = 0
    p["bo"][6] = -10

    p["Wo"][3, 1] = 2
    p["Wo"][3, 128] = -0.2

    p["Wo"][4, 0] = 2
    p["Wo"][4, 128] = 0.2

    p["Wo"][6, 2] = 100
    p["Wo"][6, 3] = 7.5
    p["Wo"][6, 129] = 8

    # Direct clicks outrank the fallback green-block click.
    p["Wc"][0, 32 + 15] = 10
    p["Wc"][0, 9] = 1

    return p
