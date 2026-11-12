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

    # Coordinate deviations for the nine spell-grid centers.
    for k, x in enumerate((25, 30, 35)):
        W1[2 * k, 17, 2, 2] = 63.0
        b1[2 * k] = -x
        W1[2 * k + 1, 17, 2, 2] = -63.0
        b1[2 * k + 1] = x

    for k, y in enumerate((50, 55, 60)):
        c = 6 + 2 * k
        W1[c, 16, 2, 2] = 63.0
        b1[c] = -y
        W1[c + 1, 16, 2, 2] = -63.0
        b1[c + 1] = y

    # Player templates, in down, up, left, right order.
    base = np.array([[10, 10], [9, 9]], dtype=np.int8)
    orientations = (
        base,
        np.rot90(base, 2),
        np.rot90(base, -1),
        np.rot90(base, 1),
    )

    for k, template in enumerate(orientations):
        c = 12 + k
        for dy in range(2):
            for dx in range(2):
                W1[c, int(template[dy, dx]), 2 + dy, 2 + dx] += 1.0
        for dy in range(-1, 3):
            for dx in range(-1, 3):
                if 0 <= dy < 2 and 0 <= dx < 2:
                    continue
                W1[c, 9, 2 + dy, 2 + dx] -= 1.0
                W1[c, 10, 2 + dy, 2 + dx] -= 1.0
        b1[c] = -3.5

        c = 16 + k
        large = np.repeat(np.repeat(template, 2, axis=0), 2, axis=1)
        for dy in range(4):
            for dx in range(4):
                W1[c, int(large[dy, dx]), 1 + dy, 1 + dx] += 1.0
        b1[c] = -15.5

    # Door-corner detectors.
    for k in range(4):
        c = 20 + k
        template = np.full((2, 2), 9, dtype=np.int8)
        template[k // 2, k % 2] = 10
        for dy in range(2):
            for dx in range(2):
                W1[c, int(template[dy, dx]), 2 + dy, 2 + dx] += 1.0
        b1[c] = -3.5

    # Conservative 2x2 wall footprint.
    for color in (3, 5):
        for dy in range(2):
            for dx in range(2):
                W1[24, color, 2 + dy, 2 + dx] = 1.0

    # Resize-button template.
    resize_icon = (
        (3, 3, 3, 3, 3),
        (3, 2, 2, 2, 15),
        (3, 2, 2, 2, 15),
        (3, 2, 2, 2, 2),
        (3, 15, 15, 2, 2),
    )
    for dy, row in enumerate(resize_icon):
        for dx, color in enumerate(row):
            W1[25, color, dy, dx] += 1.0
    b1[25] = -24.5

    W1[26, 14, 2, 2] = 1.0  # Selected grid cell.

    # Switch rings, anchored at (top-left + 1, top-left + 1).
    for channel, center_color in ((27, 13), (28, 12)):
        for dy in range(4):
            for dx in range(4):
                color = center_color if 1 <= dy <= 2 and 1 <= dx <= 2 else 6
                W1[channel, color, 1 + dy, 1 + dx] += 1.0
        b1[channel] = -15.5

    W1[29, 0, 2, 2] = 1.0
    W1[30, 17, 2, 2] = 1.0
    W1[31, 16, 2, 2] = 1.0

    # Player detections.
    for k in range(4):
        W2[k, 12 + k, 1, 1] = 2.0
        W2[4 + k, 16 + k, 2, 2] = 2.0

    # Expand each door-corner detection by one pixel.
    for k in range(4):
        for dy in range(3):
            for dx in range(3):
                W2[8 + k, 20 + k, dy, dx] = 2.0

    W2[12, 24, 1, 1] = 1.0
    W2[13, 25, 1, 1] = 2.0
    W2[14, 26, 1, 1] = 1.0

    # Exact grid-cell centers.
    for row in range(3):
        for col in range(3):
            c = 15 + 3 * row + col
            for j in (
                2 * col,
                2 * col + 1,
                6 + 2 * row,
                6 + 2 * row + 1,
            ):
                W2[c, j, 1, 1] = -1.0
            b2[c] = 1.0

    # Switch seeds expanded across nearby rows and columns.
    for switch_channel in (27, 28):
        for offset in range(3):
            W2[24, switch_channel, offset, 1] = 2.0
            W2[25, switch_channel, 1, offset] = 2.0
        W2[29, switch_channel, 1, 1] = 2.0

    W2[26, 29, 1, 1] = 1.0  # Hint-colored pixel.
    W2[27, 30, 1, 1] = 1.0  # Column.
    W2[28, 31, 1, 1] = 1.0  # Row.

    # Main visual features.
    W3[0, 0:8] = 1.0
    W3[1, 4:8] = 1.0
    W3[2, 8:12] = 1.0
    W3[3, 12] = 1.0
    W3[4, 13] = 1.0

    resize_cells = {1, 3, 5, 7}
    fire_cells = {1, 4, 7}

    # Resize and fire pattern mismatches.
    for i in range(9):
        c = 5 + i
        W3[c, 15 + i] = 1.0
        if i in resize_cells:
            W3[c, 14] = -1.0
        else:
            W3[c, 14] = 1.0
            b3[c] = -1.0

        c = 14 + i
        W3[c, 15 + i] = 1.0
        if i in fire_cells:
            W3[c, 14] = -1.0
        else:
            W3[c, 14] = 1.0
            b3[c] = -1.0

    # Player aligned with a switch, in up/down/left/right order.
    for direction in range(4):
        c = 23 + direction
        W3[c, 0:8] = 1.0
        if direction < 2:
            W3[c, 64 + 25] = 1.0
        else:
            W3[c, 32 + 24] = 1.0
        b3[c] = -1.0

    # Player and switch coordinates, gated by their detections.
    W3[27, 27] = 1.0
    W3[27, 0:8] = 2.0
    b3[27] = -2.0

    W3[28, 27] = 1.0
    W3[28, 29] = 2.0
    b3[28] = -2.0

    W3[29, 28] = 1.0
    W3[29, 0:8] = 2.0
    b3[29] = -2.0

    W3[30, 28] = 1.0
    W3[30, 29] = 2.0
    b3[30] = -2.0

    # Unselected hinted cells.
    W3[31, 15:24] = 1.0
    W3[31, 26] = 1.0
    b3[31] = -1.0

    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    # A staged max of the four neighboring navigation values.
    Wh[1, 0, 0, 1] = 1.0
    Wh[1, 0, 2, 1] = -1.0
    Wh[2, 0, 1, 0] = 1.0
    Wh[2, 0, 1, 2] = -1.0
    Wh[3, 0, 2, 1] = 1.0
    Wh[4, 0, 1, 2] = 1.0

    Wh[5, 1, 1, 1] = 1.0
    Wh[5, 3, 1, 1] = 1.0
    Wh[6, 2, 1, 1] = 1.0
    Wh[6, 4, 1, 1] = 1.0

    Wh[7, 5, 1, 1] = 1.0
    Wh[7, 6, 1, 1] = -1.0
    Wh[8, 6, 1, 1] = 1.0

    # max(neighbors, goal), with a step cost and wall penalty.
    Wh[9, 7, 1, 1] = 1.0
    Wh[9, 8, 1, 1] = 1.0
    Uh[9, 2] = -100.0

    Wh[0, 9, 1, 1] = 1.0
    Uh[0, 2] = 100.0
    Uh[0, 3] = -10000.0
    bh[0] = -1.0

    # Read the navigation field two pixels away, at the player.
    Wh[10, 1, 0, 1] = 1.0
    Wh[10, 3, 0, 1] = 1.0
    Wh[11, 3, 2, 1] = 1.0
    Wh[12, 2, 1, 0] = 1.0
    Wh[12, 4, 1, 0] = 1.0
    Wh[13, 4, 1, 2] = 1.0
    for c in range(10, 14):
        Uh[c, 0] = 10000.0
        bh[c] = -10000.0

    # Diffuse switch presence; retain it at fire-pattern mismatches.
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        Wh[14, 14, 1 + dy, 1 + dx] = 0.249
    Uh[14, 28] = 1.0
    Uh[14, 30] = 1.0

    Wh[15, 14, 1, 1] = 1.0
    Uh[15, 14:23] = 1.0
    bh[15] = -1.0

    Wz, bz = p["Wz"], p["bz"]

    for direction in range(4):
        Wz[direction, 10 + direction] = 1.0

    # Resize request when the player is large and a resize button exists.
    Wz[4, 32 + 1] = 1.0
    Wz[4, 32 + 4] = 1.0
    bz[4] = -1.5

    # Directional switch alignment. Two ReLUs per direction form a
    # clipped indicator by subtraction in the output and memory heads.
    coordinate_pairs = (
        (29, 30),  # player row - switch row
        (30, 29),
        (27, 28),  # player column - switch column
        (28, 27),
    )
    epsilon = 1.0 / 63.0
    for direction, (positive, negative) in enumerate(coordinate_pairs):
        for extra in range(2):
            c = 5 + 2 * direction + extra
            Wz[c, 32 + positive] = 1.0
            Wz[c, 32 + negative] = -1.0
            Wz[c, 32 + 23 + direction] = 1.0
            bz[c] = -1.0 - epsilon * (1 + extra)

    # Direction latches. Clicking does not clear the last direction.
    for direction in range(4):
        p["Wm"][direction, direction] = 3.0
        for other in range(4):
            p["Wma"][direction, 1 + other] = 5.0 if other == direction else -5.0

    # Fire-ready flags combine alignment with the latched or immediately
    # previous direction.
    for direction in range(4):
        c = 4 + direction
        p["Wm"][c, direction] = 5.0
        for other in range(4):
            p["Wma"][c, 1 + other] = 10.0 if other == direction else -10.0
        p["Wmz"][c, 5 + 2 * direction] = 630.0
        p["Wmz"][c, 6 + 2 * direction] = -630.0
        p["bm"][c] = -12.0

    p["bo"][0] = -1000000.0
    for direction in range(4):
        action = 1 + direction
        p["Wo"][action, direction] = 1.0
        p["Wo"][action, 5 + 2 * direction] = 18900.0
        p["Wo"][action, 6 + 2 * direction] = -18900.0
        p["Wo"][action, 128 + direction] = 2.0
        p["bo"][action] = 1.0

    # Four normally-negative fire flags contribute a click only when
    # one direction is both aligned and selected.
    p["bo"][6] = 1500.0
    for direction in range(4):
        p["Wo"][6, 128 + 4 + direction] = 500.0
    p["Wo"][6, 4] = 100000.0

    # Click hints first; otherwise choose a pattern mismatch.
    p["Wc"][0, 31] = 100.0
    for i in range(9):
        p["Wc"][0, 5 + i] = 1.0 if i in resize_cells else 2.0
    p["Wc"][0, 32 + 15] = 100000000.0

    return p
