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

    colors = (6, 8, 9, 10, 11, 12, 14)
    ncolors = len(colors)
    diagonals = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    neighbors = tuple(
        (dy, dx)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if dy != 0 or dx != 0
    )

    # Paired threshold units detect goal centers and locally identifiable
    # selected centers.
    for i, color in enumerate(colors):
        ga, gb = 2 * i, 2 * i + 1

        if color == 8:
            # Allow two marker-border pixels to be covered by a shape.
            for unit in (ga, gb):
                p["W1"][unit, color, 2, 2] = 32.0
                for dy, dx in neighbors:
                    p["W1"][unit, 4, 2 + dy, 2 + dx] = 2.0
            p["b1"][ga] = -43.0
            p["b1"][gb] = -44.0
        else:
            for unit in (ga, gb):
                p["W1"][unit, color, 2, 2] = 16.0
                for dy, dx in diagonals:
                    p["W1"][unit, 4, 2 + dy, 2 + dx] = 1.0
            p["b1"][ga] = -19.0
            p["b1"][gb] = -20.0

        sa = 2 * ncolors + 2 * i
        sb = sa + 1
        for unit in (sa, sb):
            p["W1"][unit, 0, 2, 2] = 16.0
            for dy, dx in neighbors:
                p["W1"][unit, color, 2 + dy, 2 + dx] = 1.0
        p["b1"][sa] = -16.0
        p["b1"][sb] = -17.0

        p["W2"][i, ga, 1, 1] = 1.0
        p["W2"][i, gb, 1, 1] = -1.0
        p["W2"][ncolors + i, sa, 1, 1] = 1.0
        p["W2"][ncolors + i, sb, 1, 1] = -1.0

    # Pass coordinates, a color-8 plane, and the zero-center plane.
    row1 = 4 * ncolors
    col1 = row1 + 1
    color8_1 = row1 + 2
    zero1 = row1 + 3

    p["W1"][row1, 16, 2, 2] = 1.0
    p["W1"][col1, 17, 2, 2] = 1.0
    p["W1"][color8_1, 8, 2, 2] = 1.0
    p["W1"][zero1, 0, 2, 2] = 1.0

    row2 = 2 * ncolors
    col2 = row2 + 1
    color8_2 = row2 + 2
    zero2 = row2 + 3

    p["W2"][row2, row1, 1, 1] = 1.0
    p["W2"][col2, col1, 1, 1] = 1.0
    p["W2"][color8_2, color8_1, 1, 1] = 1.0
    p["W2"][zero2, zero1, 1, 1] = 1.0

    # Per color: masked goal row, column, and goal count.
    for i in range(ncolors):
        row = 3 * i
        col = row + 1
        mask = row + 2

        p["W3"][row, i] = 1.0
        p["W3"][row, row2] = 1.0
        p["b3"][row] = -1.0

        p["W3"][col, i] = 1.0
        p["W3"][col, col2] = 1.0
        p["b3"][col] = -1.0

        p["W3"][mask, i] = 1.0

    selected_row = 3 * ncolors
    selected_col = selected_row + 1
    selected_flags = selected_row + 2
    color8_plane = selected_flags + ncolors
    zero_plane = color8_plane + 1

    p["W3"][selected_row, zero2] = 1.0
    p["W3"][selected_row, row2] = 1.0
    p["b3"][selected_row] = -1.0

    p["W3"][selected_col, zero2] = 1.0
    p["W3"][selected_col, col2] = 1.0
    p["b3"][selected_col] = -1.0

    for i in range(ncolors):
        p["W3"][selected_flags + i, ncolors + i] = 1.0

    p["W3"][color8_plane, color8_2] = 1.0
    p["W3"][zero_plane, zero2] = 1.0

    # Diffuse color 8 and read it at an isolated zero center. Two
    # thresholded channels encode a capped binary-like fallback signal.
    for dy, dx in neighbors:
        p["Wh"][0, 0, 1 + dy, 1 + dx] = 0.1225
    p["Uh"][0, color8_plane] = 0.1

    p["Wh"][1, 0, 1, 1] = 1.0
    p["Uh"][1, zero_plane] = 10.0
    for i in range(ncolors):
        p["Uh"][1, selected_flags + i] = -10.0
    p["bh"][1] = -10.0

    p["Wh"][2, 1, 1, 1] = 1.0
    p["bh"][2] = -0.03
    p["Wh"][3, 1, 1, 1] = 1.0
    p["bh"][3] = -0.04

    # g offsets: h max, h mean, f3 max, f3 mean.
    f3_max = 32
    f3_mean = 64
    count_gain = 20.0
    color_penalty = 5000.0
    next_unit = 0

    def make_gate(color_index, threshold, axis=None, count=None):
        nonlocal next_unit
        unit = next_unit
        next_unit += 1

        goal_mask = 3 * color_index + 2
        p["Wz"][unit, f3_mean + goal_mask] = 4096.0 * count_gain
        p["Wz"][unit, f3_max + selected_flags + color_index] = (
            color_penalty
        )
        p["bz"][unit] = (
            count_gain * (0.5 - threshold) - color_penalty
        )

        if colors[color_index] == 8:
            # h2 - h3 is approximately 0.01 for a sufficiently strong
            # color-8 signal at a center with no local color detection.
            p["Wz"][unit, 2] += color_penalty / 0.01
            p["Wz"][unit, 3] -= color_penalty / 0.01

        if axis is not None:
            goal_coordinate = 3 * color_index + axis
            chosen_coordinate = selected_row if axis == 0 else selected_col
            p["Wz"][unit, f3_mean + goal_coordinate] += 4096.0
            p["Wz"][unit, f3_max + chosen_coordinate] -= float(count)

        return unit

    for i, color in enumerate(colors):
        if color == 8:
            counts = tuple(range(2, 9))
        elif color == 6:
            counts = (2, 3)
        else:
            counts = (2, 3, 4)

        thresholds = sorted(set(counts) | {n + 1 for n in counts})
        baselines = {
            threshold: make_gate(i, threshold)
            for threshold in thresholds
        }

        for count in counts:
            scale = 4.0 / count
            for axis in (0, 1):
                lower = make_gate(i, count, axis, count)
                upper = make_gate(i, count + 1, axis, count)

                negative_action = 1 if axis == 0 else 3
                positive_action = 2 if axis == 0 else 4

                for action, sign in (
                    (positive_action, 1.0),
                    (negative_action, -1.0),
                ):
                    factor = sign * scale
                    p["Wo"][action, lower] += factor
                    p["Wo"][action, upper] -= factor
                    p["Wo"][action, baselines[count]] -= factor
                    p["Wo"][action, baselines[count + 1]] += factor

    p["bo"][5] = 0.075
    return p
