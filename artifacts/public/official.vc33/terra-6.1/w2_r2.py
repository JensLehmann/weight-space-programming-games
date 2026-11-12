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
    p = {name: np.zeros(shape, dtype=np.float32)
         for name, shape in shapes.items()}

    W1, b1 = p["W1"], p["b1"]
    W2, b2 = p["W2"], p["b2"]
    W3 = p["W3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    # Find the upper-left part of a player or target marker.
    offsets = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-2, 0), (2, 0), (0, -2), (0, 2),
    ]
    for group, neighbor_color in enumerate((4, 5)):
        for i, (dy, dx) in enumerate(offsets):
            out = 8 * group + i
            for color in (11, 14):
                W1[out, color, 2, 2] += 1
                W1[out, color, 1, 2] -= 1
                W1[out, color, 2, 1] -= 1
            W1[out, neighbor_color, 2 + dy, 2 + dx] += 1
            b1[out] = -1

    # Yellow pixels adjacent to a wall, by side.
    adjacent = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i, (dy, dx) in enumerate(adjacent):
        out = 16 + i
        W1[out, 9, 2, 2] = 1
        W1[out, 5, 2 + dy, 2 + dx] = 1
        b1[out] = -1

    for out, color in ((20, 5), (21, 1), (22, 12),
                       (23, 9), (26, 0), (27, 3)):
        W1[out, color, 2, 2] = 1

    # Long-axis tests for open gates.
    W1[24, 12, 2, 2] = 1
    W1[24, 12, 0, 2] = 1
    W1[24, 12, 4, 2] = 1
    b1[24] = -2

    W1[25, 12, 2, 2] = 1
    W1[25, 12, 2, 0] = 1
    W1[25, 12, 2, 4] = 1
    b1[25] = -2

    # Clip marker-detector sums using differences of ReLUs.
    for i in range(8):
        W2[0, i, 1, 1] = 1
        W2[1, i, 1, 1] = 1
        W2[2, 8 + i, 1, 1] = 1
        W2[3, 8 + i, 1, 1] = 1
    b2[1] = -1
    b2[3] = -1

    for i in range(4):
        W2[4 + i, 16 + i, 1, 1] = 1
    for i in range(6):
        W2[8 + i, 20 + i, 1, 1] = 1
    W2[14, 26, 1, 1] = 1
    W2[15, 27, 1, 1] = 1

    # Axial broadcasts of player and target marker rows and columns.
    W3[0, 32 + 0] = 1
    W3[0, 32 + 1] = -1
    W3[1, 64 + 0] = 1
    W3[1, 64 + 1] = -1
    W3[2, 32 + 2] = 1
    W3[2, 32 + 3] = -1
    W3[3, 64 + 2] = 1
    W3[3, 64 + 3] = -1

    # Wall, control, gate, floor, and background masks.
    W3[4, 8] = 1
    W3[4, 9] = 1
    W3[4, 10] = 1
    W3[5, 4] = 1
    W3[5, 5] = 1
    W3[6, 6] = 1
    W3[6, 7] = 1
    W3[7, 10] = 1
    W3[8, 11] = 1
    W3[9, 12] = 1
    W3[10, 13] = 1
    W3[11, 4] = 1       # Yellow immediately below a horizontal wall.
    W3[12, 14] = 1      # Floor.
    W3[13, 15] = 1      # Background.
    W3[14, 11] = 1      # Yellow.

    def shifted(out, inp, dy, dx, weight=1):
        Wh[out, inp, 1 + dy, 1 + dx] += weight

    # Player rays: right, left, down, up. Walls stop propagation.
    ray_shifts = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    for i, (dy, dx) in enumerate(ray_shifts):
        shifted(i, i, dy, dx)
        Uh[i, 1 if i < 2 else 0] = 1
        Uh[i, 4] = -20

    # Target rays in the same directions.
    for i, (dy, dx) in enumerate(ray_shifts):
        shifted(4 + i, 4 + i, dy, dx)
        Uh[4 + i, 3 if i < 2 else 2] = 1

    # A floor signal travels downward through floor and walls but is
    # extinguished by background or yellow pixels.
    shifted(14, 14, -1, 0, 0.98)
    Uh[14, 12] = 0.1
    Uh[14, 13] = -10
    Uh[14, 14] = -10

    # At a control below a wall, sample the donor signal diagonally
    # across the edge of the yellow control.
    shifted(15, 14, -1, -1)
    shifted(15, 14, -1, 1)
    Uh[15, 11] = 10
    bh[15] = -10.02

    # Prefer an available horizontal-wall control in the player's lane.
    Wh[8, 2, 1, 1] = 1
    Wh[8, 3, 1, 1] = 1
    Wh[8, 15, 1, 1] = 10
    bh[8] = -1

    # Original lane heuristic for vertical-wall controls.
    Wh[9, 0, 1, 1] = 1
    Wh[9, 1, 1, 1] = 1
    Uh[9, 6] = 20
    bh[9] = -20.5

    # Open-gate tests for player and target on opposite sides.
    gate_tests = [
        (0, (0, -1), 5, (0, 1), 9),
        (1, (0, 1), 4, (0, -1), 9),
        (2, (-1, 0), 7, (1, 0), 10),
        (3, (1, 0), 6, (-1, 0), 10),
    ]
    for j, (pr, po, gr, go, axis) in enumerate(gate_tests):
        out = 10 + j
        shifted(out, pr, po[0], po[1])
        shifted(out, gr, go[0], go[1])
        Uh[out, axis] = 20
        bh[out] = -21.5

    p["bo"][6] = 1

    # Yellow is a fallback. Available donors, same-lane controls, and
    # suitable open gates receive progressively stronger scores.
    p["Wc"][0, 8] = 2
    p["Wc"][0, 32 + 15] = 2
    p["Wc"][0, 32 + 8] = 10
    p["Wc"][0, 32 + 9] = 10
    for i in range(10, 14):
        p["Wc"][0, 32 + i] = 30

    return p
