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

    groups = ((8,), (9,), (11,), (12,), (10,), (6, 7, 14))

    # Target markers and local selected-color evidence.
    for i, colors in enumerate(groups):
        for color in colors:
            p["W1"][i, color, 2, 2] = 16.0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx or dy:
                    p["W1"][i, 4, 2 + dy, 2 + dx] = 1.0
        p["b1"][i] = -23.0

        local = 6 + i
        p["W1"][local, 0, 2, 2] = 32.0
        for color in colors:
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dx or dy:
                        p["W1"][local, color, 2 + dy, 2 + dx] = 1.0
        p["b1"][local] = -32.0

    p["W1"][12, 16, 2, 2] = 1.0
    p["W1"][13, 17, 2, 2] = 1.0

    # Solid interiors of color-8 and color-9 paint stations.
    for out, color in ((14, 8), (15, 9)):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                p["W1"][out, color, 2 + dy, 2 + dx] = 1.0
        p["b1"][out] = -8.0

    # Isolated zero, potentially the center of a hollow piece.
    p["W1"][16, 0, 2, 2] = 32.0
    for color in (1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 14):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx or dy:
                    p["W1"][16, color, 2 + dy, 2 + dx] = -1.0
    p["b1"][16] = -31.0

    p["W1"][17, 8, 2, 2] = 1.0
    p["W1"][18, 12, 2, 2] = 1.0
    p["W1"][18, 14, 2, 2] = 1.0

    # Selected color-8 horizontal line: horizontal neighbors but
    # no color-8 diagonal neighbors.
    p["W1"][19, 0, 2, 2] = 8.0
    p["W1"][19, 8, 2, 1] = 2.0
    p["W1"][19, 8, 2, 3] = 2.0
    for dy in (-1, 1):
        for dx in (-1, 1):
            p["W1"][19, 8, 2 + dy, 2 + dx] = -2.0
    p["b1"][19] = -11.0

    # W2 binarizes local selected-color evidence.
    for i in range(len(groups)):
        p["W2"][i, i, 1, 1] = 1.0
        p["W2"][6 + 2 * i, 6 + i, 1, 1] = 1.0
        p["W2"][7 + 2 * i, 6 + i, 1, 1] = 1.0
        p["b2"][7 + 2 * i] = -1.0

    # If the zero has color-8 neighbors, avoid simultaneously assigning
    # nearby color-12 pixels to the selected piece.
    p["W2"][12, 6, 1, 1] = -25.0
    p["W2"][13, 6, 1, 1] = -25.0

    p["W2"][18, 12, 1, 1] = 1.0
    p["W2"][19, 13, 1, 1] = 1.0
    p["W2"][20, 14, 1, 1] = 1.0
    p["W2"][21, 15, 1, 1] = 1.0
    p["W2"][22, 16, 1, 1] = 1.0
    p["W2"][23, 17, 1, 1] = 1.0
    p["W2"][24, 18, 1, 1] = 1.0
    p["W2"][26, 19, 1, 1] = 1.0

    # Axial tests for hollow centers.
    for out, color_channel in ((30, 23), (31, 24)):
        p["W3"][out, 22] = 1.0
        p["W3"][out, 32 + color_channel] = 1.0
        p["W3"][out, 64 + color_channel] = 1.0
        p["b3"][out] = -2.0

    hollow = {0: 30, 3: 31, 5: 31}

    # Four f3 channels per color group: marker, row encoding,
    # column encoding, and local selection flag.
    for i in range(len(groups)):
        base = 4 * i
        local_pos = 6 + 2 * i
        local_neg = 7 + 2 * i
        p["W3"][base, i] = 1.0

        for out, coord in ((base + 1, 18), (base + 2, 19)):
            p["W3"][out, i] = 2.0
            p["W3"][out, local_pos] = 5.0
            p["W3"][out, local_neg] = -5.0
            if i in hollow:
                p["W3"][out, 22] = 1.0
            p["W3"][out, coord] = 1.0
            p["b3"][out] = -1.0

        p["W3"][base + 3, local_pos] = 1.0
        p["W3"][base + 3, local_neg] = -1.0

        p["Uh"][2 * i, base + 1] = 1.0
        p["Uh"][2 * i + 1, base + 2] = 1.0
        if i in hollow:
            p["Uh"][2 * i, hollow[i]] = 4.0
            p["Uh"][2 * i + 1, hollow[i]] = 4.0
        p["bh"][2 * i] = -3.0
        p["bh"][2 * i + 1] = -3.0

    # Paint-station coordinates. Channel 29 is reserved for line type.
    for station, f2_channel in enumerate((20, 21)):
        base = 24 + 3 * station
        for out, coord in ((base, 18), (base + 1, 19)):
            p["W3"][out, f2_channel] = 2.0
            p["W3"][out, coord] = 1.0
            p["b3"][out] = -1.0
        if station == 0:
            p["W3"][base + 2, f2_channel] = 1.0

    p["W3"][29, 26] = 1.0

    # Recurrent horizontal target accumulation for color 8.
    # Channels 12 and 13 propagate marker evidence in opposite directions.
    p["Uh"][12, 0] = 1.0
    p["Uh"][13, 0] = 1.0
    p["Wh"][12, 12, 1, 0] = 0.995
    p["Wh"][13, 13, 1, 2] = 0.995

    # Channel 14 scores rows; channel 15 adds a small row-coordinate
    # term at markers, allowing a row estimate from their maxima.
    for out in (14, 15):
        p["Wh"][out, 12, 1, 1] = 1.0
        p["Wh"][out, 13, 1, 1] = 1.0
        p["Uh"][out, 0] = -1.0
    p["Uh"][15, 1] = 0.01
    p["Uh"][15, 3] = -0.05
    p["Uh"][15, 30] = -0.01

    # g layout: h maxima, h means, f3 maxima, f3 means.
    z_index = 0
    gate = 32.0

    for i in range(len(groups)):
        base = 4 * i
        hrow = 2 * i
        hcol = 2 * i + 1
        selected = 32 + base + 3
        iso = 32 + hollow[i] if i in hollow else None
        count_mean = 64 + base
        row_mean = 64 + base + 1
        col_mean = 64 + base + 2

        goal_row = z_index
        goal_col = z_index + 1
        z_index += 2

        for unit, moment, selected_coord in (
            (goal_row, row_mean, hrow),
            (goal_col, col_mean, hcol),
        ):
            p["Wz"][unit, moment] = 4096.0
            p["Wz"][unit, count_mean] = -4096.0
            p["Wz"][unit, selected_coord] = -1.0
            p["Wz"][unit, selected] = gate - 3.0
            if iso is not None:
                p["Wz"][unit, iso] = gate + 1.0
            p["bz"][unit] = -gate

        p["Wo"][2, goal_row] += 1.0
        p["Wo"][1, goal_row] -= 1.0
        p["Wo"][4, goal_col] += 1.0
        p["Wo"][3, goal_col] -= 1.0

        # Count-times-coordinate construction, exact up to this limit.
        limit = 9 if i in (0, 1) else (2 if i == 4 else 4)
        for k in range(1, limit + 1):
            baseline = z_index
            row_term = z_index + 1
            col_term = z_index + 2
            z_index += 3

            for unit in (baseline, row_term, col_term):
                p["Wz"][unit, count_mean] = 4096.0
                p["bz"][unit] = -float(k)

            p["Wz"][row_term, hrow] = 1.0
            p["Wz"][row_term, selected] = -1.0
            p["Wz"][col_term, hcol] = 1.0
            p["Wz"][col_term, selected] = -1.0
            if iso is not None:
                p["Wz"][row_term, iso] = -1.0
                p["Wz"][col_term, iso] = -1.0

            p["Wo"][2, row_term] -= 1.0
            p["Wo"][2, baseline] += 1.0
            p["Wo"][1, row_term] += 1.0
            p["Wo"][1, baseline] -= 1.0

            p["Wo"][4, col_term] -= 1.0
            p["Wo"][4, baseline] += 1.0
            p["Wo"][3, col_term] += 1.0
            p["Wo"][3, baseline] -= 1.0

    # Heuristic routes to visible paint stations when the current color
    # has no visible matching marker.
    routes = ((3, 1), (4, 0), (2, 0), (5, 1))
    for group, station in routes:
        selected = 32 + 4 * group + 3
        iso = 32 + hollow[group] if group in hollow else None
        own_marker = 32 + 4 * group
        station_base = 24 + 3 * station

        # Station 1's row-coordinate feature also serves as existence
        # evidence because channel 29 is used for line detection.
        station_flag = 32 + (26 if station == 0 else 27)

        for axis in range(2):
            selected_coord = 2 * group + axis
            station_coord = 32 + station_base + axis

            for sign in (1.0, -1.0):
                unit = z_index
                z_index += 1
                p["Wz"][unit, station_coord] = sign
                p["Wz"][unit, selected_coord] = -sign
                p["Wz"][unit, selected] = gate
                if iso is not None:
                    p["Wz"][unit, iso] = gate
                p["Wz"][unit, own_marker] = -gate
                p["Wz"][unit, station_flag] = gate
                p["bz"][unit] = -2.0 * gate

                if axis == 0:
                    action = 2 if sign > 0 else 1
                else:
                    action = 4 if sign > 0 else 3
                p["Wo"][action, unit] += 3.0

    # Extra row steering for a selected color-8 horizontal line.
    line_flag = 32 + 29
    selected8 = 32 + 3
    for sign, action in ((1.0, 2), (-1.0, 1)):
        unit = z_index
        z_index += 1
        p["Wz"][unit, 15] = 100.0 * sign
        p["Wz"][unit, 14] = -100.0 * sign
        p["Wz"][unit, 0] = -sign
        p["Wz"][unit, selected8] = sign
        p["Wz"][unit, line_flag] = gate
        p["bz"][unit] = -gate - sign
        p["Wo"][action, unit] += 5.0

    # If no row has strong multi-marker evidence, cycle the line.
    unit = z_index
    z_index += 1
    p["Wz"][unit, 14] = -1.0
    p["Wz"][unit, line_flag] = 4.0
    p["bz"][unit] = -2.75
    p["Wo"][5, unit] = 4.0

    # Penalize immediately reversing the previous movement.
    opposite = {1: 2, 2: 1, 3: 4, 4: 3}
    for j, action in enumerate((1, 2, 3, 4)):
        p["Wma"][j, opposite[action]] = 20.0
        p["bm"][j] = -10.0
        p["Wo"][action, 128 + j] = -0.5
        p["bo"][action] = -0.5

    p["bo"][5] = 0.005
    return p
