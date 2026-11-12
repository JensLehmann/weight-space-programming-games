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
    W2, b2 = p["W2"], p["b2"]
    W3, b3 = p["W3"], p["b3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]
    Wz, bz = p["Wz"], p["bz"]
    Wm, Wmz, Wma, bm = p["Wm"], p["Wmz"], p["Wma"], p["bm"]
    Wo, bo, Wc = p["Wo"], p["bo"], p["Wc"]

    # Player templates, anchored one pixel down/right from top-left.
    small = np.array([[10, 10], [9, 9]], dtype=np.int32)
    large = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)

    for k in range(4):
        pattern = np.rot90(small, k)
        for y in range(2):
            for x in range(2):
                W1[k, pattern[y, x], y + 1, x + 1] = 1.0
        for dy in range(-2, 2):
            for dx in range(-2, 2):
                if -1 <= dy <= 0 and -1 <= dx <= 0:
                    continue
                W1[k, 9, dy + 2, dx + 2] = -1.0
                W1[k, 10, dy + 2, dx + 2] = -1.0
        b1[k] = -3.5

    # Do not penalize pixels outside the large-player template:
    # the player can be directly adjacent to the goal.
    for k in range(4):
        pattern = np.rot90(large, k)
        for y in range(4):
            for x in range(4):
                W1[4 + k, pattern[y, x], y + 1, x + 1] = 1.0
        b1[4 + k] = -15.5

    # Goal corner templates.
    corner = np.full((5, 5), 10, dtype=np.int32)
    corner[0, :] = 9
    corner[:, 0] = 9
    for k in range(4):
        pattern = np.rot90(corner, k)
        for y in range(5):
            for x in range(5):
                W1[8 + k, pattern[y, x], y, x] = 1.0
        b1[8 + k] = -24.5

    # Approximate obstacle map for a small player.
    for color in (3, 5):
        for dy in range(-1, 1):
            for dx in range(-1, 1):
                W1[12, color, dy + 2, dx + 2] = 1.0

    # A 5x2 segment of the fire icon: two 6 rows, a 2 row,
    # then two more 6 rows.
    for y in range(5):
        color = 2 if y == 2 else 6
        for x in (1, 2):
            W1[13, color, y, x] = 1.0
    b1[13] = -9.5

    W1[14, 16, 2, 2] = 1.0
    W1[15, 17, 2, 2] = 1.0
    W1[16, 14, 2, 2] = 1.0
    W1[17, 15, 2, 2] = 1.0
    W1[19, 0, 2, 2] = 1.0

    # Fire target: 4x4 colour-6 border with colour-13 centre.
    switch = np.full((4, 4), 6, dtype=np.int32)
    switch[1:3, 1:3] = 13
    for y in range(4):
        for x in range(4):
            W1[18, switch[y, x], y + 1, x + 1] = 1.0
    b1[18] = -15.5

    rows = (50, 55, 60)
    cols = (25, 30, 35)
    for i, row in enumerate(rows):
        W1[20 + i, 16, 2, 2] = 63.0
        b1[20 + i] = -float(row)
        W1[23 + i, 16, 2, 2] = -63.0
        b1[23 + i] = float(row)
    for i, col in enumerate(cols):
        W1[26 + i, 17, 2, 2] = 63.0
        b1[26 + i] = -float(col)
        W1[29 + i, 17, 2, 2] = -63.0
        b1[29 + i] = float(col)

    # Player, goal, clearance and orientation.
    for k in range(4):
        W2[0, k, 1, 1] = 2.0
        W2[1, 4 + k, 1, 1] = 2.0
        W2[2, 8 + k, 1, 1] = 2.0
        W2[20 + k, k, 1, 1] = 2.0
        W2[20 + k, 4 + k, 1, 1] = 2.0

    W2[3, 12, 1, 1] = -1.0
    b2[3] = 1.0
    W2[4, 14, 1, 1] = 1.0
    W2[5, 15, 1, 1] = 1.0
    W2[6, 16, 1, 1] = 1.0
    W2[7, 19, 1, 1] = 1.0
    W2[8, 18, 1, 1] = 2.0

    # Fire-icon shape, with a weak preference for the left panel.
    W2[9, 13, 1, 1] = 2.0
    W2[9, 15, 1, 1] = -1.0

    # Resize-icon pixels in either canonical panel location.
    W2[24, 17, 1, 1] = 1.0
    W2[24, 15, 1, 1] = -63.0 / 30.0
    W2[24, 21, 1, 1] = -1.0 / 12.0
    W2[24, 24, 1, 1] = -1.0 / 12.0

    W2[25, 17, 1, 1] = 1.0
    W2[25, 14, 1, 1] = -63.0 / 15.0
    W2[25, 15, 1, 1] = -63.0 / 15.0

    # Exact spell-grid click coordinates.
    for r in range(3):
        for c in range(3):
            out = 11 + 3 * r + c
            b2[out] = 1.0
            W2[out, 20 + r, 1, 1] = -1.0
            W2[out, 23 + r, 1, 1] = -1.0
            W2[out, 26 + c, 1, 1] = -1.0
            W2[out, 29 + c, 1, 1] = -1.0

    W3[0, 0] = 1.0
    W3[1, 1] = 1.0
    W3[2, 2] = 1.0
    W3[3, 3] = -1.0
    b3[3] = 1.0

    # Player coordinates.
    for out, coord in ((4, 4), (5, 5)):
        W3[out, 0] = 1.0
        W3[out, 1] = 1.0
        W3[out, coord] = 1.0
        b3[out] = -1.0

    # Selected cell and black hint cell.
    for cell in range(9):
        W3[6, 11 + cell] = 1.0
        W3[7, 11 + cell] = 1.0
    W3[6, 6] = 1.0
    W3[7, 7] = 1.0
    b3[6] = -1.0
    b3[7] = -1.0

    W3[8, 9] = 1.0
    W3[9, 24] = 1.0
    W3[9, 25] = 1.0

    # Goal AND approximate obstacle.
    W3[10, 2] = 1.0
    W3[10, 3] = 1.0
    b3[10] = -1.0

    # Fire target aligned with player row.
    target_rowmax = 32 + 8
    W3[11, 0] = 1.0
    W3[11, 1] = 1.0
    W3[11, target_rowmax] = 1.0
    b3[11] = -1.0

    # k=3 faces left; k=1 faces right.
    W3[12, 23] = 1.0
    W3[12, target_rowmax] = 1.0
    b3[12] = -1.0
    W3[13, 21] = 1.0
    W3[13, target_rowmax] = 1.0
    b3[13] = -1.0

    # Target column, gated by target detection.
    W3[14, 8] = 1.0
    W3[14, 5] = 1.0
    b3[14] = -1.0

    # Cross-hint side cells distinguish resize from the vertical fire hint.
    W3[15, 14] = 1.0
    W3[15, 16] = 1.0
    W3[15, 7] = 1.0
    b3[15] = -1.0

    # Tied block: max of four neighboring goal-field values.
    Wh[1, 0, 0, 1] = 1.0
    Wh[1, 0, 2, 1] = -1.0
    Wh[2, 0, 2, 1] = 1.0
    Wh[3, 0, 1, 0] = 1.0
    Wh[3, 0, 1, 2] = -1.0
    Wh[4, 0, 1, 2] = 1.0

    Wh[5, 1, 1, 1] = 1.0
    Wh[5, 2, 1, 1] = 1.0
    Wh[5, 3, 1, 1] = -1.0
    Wh[5, 4, 1, 1] = -1.0
    Wh[6, 3, 1, 1] = 1.0
    Wh[6, 4, 1, 1] = 1.0

    # Fixed source: max(neighbor_max, goal_seed).
    Wh[7, 5, 1, 1] = 1.0
    Wh[7, 6, 1, 1] = 1.0
    Wh[7, 8, 1, 1] = -1.0
    Uh[8, 2] = 100.0

    Wh[0, 7, 1, 1] = 1.0
    Wh[0, 8, 1, 1] = 1.0
    Uh[0, 3] = -100.0
    Uh[0, 10] = 100.0
    bh[0] = -1.0

    # Read neighboring field values at player anchors.
    offsets = ((0, 1), (2, 1), (1, 0), (1, 2))
    for i, (ky, kx) in enumerate(offsets):
        ch = 9 + i
        Wh[ch, 0, ky, kx] = 1.0
        Uh[ch, 0] = 100.0
        Uh[ch, 1] = 100.0
        bh[ch] = -100.0

    # g = [h.max, h.mean, f3.max, f3.mean].
    for i in range(4):
        Wz[i, 9 + i] = 1.0
    Wz[4, 32 + 4] = 1.0
    Wz[5, 32 + 5] = 1.0
    Wz[6, 32 + 9] = 1.0
    Wz[7, 32 + 1] = 1.0
    Wz[8, 32 + 6] = 1.0
    Wz[13, 32 + 15] = 1.0
    Wz[14, 32 + 7] = 1.0

    # Target to left/right.
    Wz[9, 32 + 11] = 2.0
    Wz[9, 32 + 5] = 1.0
    Wz[9, 32 + 14] = -1.0
    bz[9] = -2.0

    Wz[10, 32 + 11] = 2.0
    Wz[10, 32 + 14] = 1.0
    Wz[10, 32 + 5] = -1.0
    bz[10] = -2.0

    # Aligned and facing target.
    Wz[11, 32 + 12] = 2.0
    Wz[11, 32 + 5] = 1.0
    Wz[11, 32 + 14] = -1.0
    bz[11] = -2.0

    Wz[12, 32 + 13] = 2.0
    Wz[12, 32 + 14] = 1.0
    Wz[12, 32 + 5] = -1.0
    bz[12] = -2.0

    # Remember position and compare with the preceding observation.
    Wmz[0, 4] = 0.1
    Wmz[1, 5] = 0.1
    slope = 6000.0
    epsilon = 0.0012

    for plus, minus, old, current in ((2, 3, 0, 4), (4, 5, 1, 5)):
        Wm[plus, old] = -slope
        Wm[minus, old] = -slope
        Wmz[plus, current] = slope * 0.1
        Wmz[minus, current] = slope * 0.1
        bm[plus] = slope * epsilon
        bm[minus] = -slope * epsilon

    Wma[6, 1:5] = 4.0
    bm[6] = -2.0

    # Large player and resize icon both present.
    Wmz[7, 7] = 20.0
    Wmz[7, 6] = 20.0
    bm[7] = -25.0

    # Continue drawing after a click when a cell is selected.
    Wmz[8, 8] = 6.0
    Wma[8, 6] = 6.0
    bm[8] = -9.0

    # Previous-direction and blocked-direction indicators.
    for i in range(4):
        flag = 9 + i
        plus = 13 + 2 * i
        minus = plus + 1
        old = 0 if i < 2 else 1
        current = 4 if i < 2 else 5
        action = 1 + i

        Wma[flag, action] = 5.0
        Wm[plus, old] = -slope
        Wm[minus, old] = slope
        Wmz[plus, current] = slope * 0.1
        Wmz[minus, current] = -slope * 0.1
        Wma[plus, action] = 8.0
        Wma[minus, action] = 8.0
        bm[plus] = slope * epsilon - 8.0
        bm[minus] = slope * epsilon - 8.0

    # Ready to fire; suppress immediate repeated icon clicks.
    Wmz[22, 11] = 400.0
    Wmz[22, 12] = 400.0
    Wma[22, 6] = -8.0
    bm[22] = -4.0

    # A large player with a visible resize cross hint.
    Wmz[23, 7] = 10.0
    Wmz[23, 13] = 10.0
    bm[23] = -15.0

    # Continue from an icon click when its hint has appeared.
    Wmz[24, 14] = 8.0
    Wma[24, 6] = 8.0
    bm[24] = -12.0

    base = 128
    for i in range(4):
        action = 1 + i
        flag = 9 + i
        plus = 13 + 2 * i
        minus = plus + 1
        Wo[action, i] = 0.2
        Wo[action, base + flag] = 4.0
        Wo[action, base + plus] = -2.0
        Wo[action, base + minus] = -2.0
        bo[action] = -4.0

    # Turn toward an aligned target.
    Wo[3, 9] = 100.0
    Wo[4, 10] = 100.0

    # The four signed gates are normally -1. Their positive transitions
    # each add 120 to the click logit.
    for mem_index in (8, 22, 23, 24):
        Wo[6, base + mem_index] = 60.0
    bo[6] = 235.0

    # Prefer an unfilled hint, then the fire icon, then resize icon.
    Wc[0, 7] = 4.0
    Wc[0, 8] = 1.0
    Wc[0, 9] = 0.5

    bo[0] = -100.0
    bo[5] = -100.0
    bo[7] = -100.0

    return p
