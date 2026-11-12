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

    colours = (14, 15)
    neighbours = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),             (0, 1),
        (1, -1),  (1, 0),    (1, 1),
    ]

    # W1[0:2]: target-ring centres.
    for group, colour in enumerate(colours):
        for dy in (-2, 2):
            for dx in (-2, 2):
                p["W1"][group, colour, 2 + dy, 2 + dx] = 1
        p["b1"][group] = -3.5

    # W1[2:4]: ball centres, identified by colour 6 and their outline.
    for group, colour in enumerate(colours):
        channel = 2 + group
        p["W1"][channel, 6, 2, 2] = 1
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            p["W1"][channel, colour, 2 + dy, 2 + dx] = 1
        p["b1"][channel] = -4.5

    # W1[4:6] and W1[6:8]: selected and unselected handles.
    for group, colour in enumerate(colours):
        for base, outline in ((4, 0), (6, 3)):
            channel = base + group
            p["W1"][channel, colour, 2, 2] = 1
            for dy, dx in neighbours:
                p["W1"][channel, outline, 2 + dy, 2 + dx] = 1
            p["b1"][channel] = -8.5

    p["W1"][8, 16, 2, 2] = 1
    p["W1"][9, 17, 2, 2] = 1

    # Visible walls under a prospective handle.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if abs(dy) + abs(dx) <= 2:
                p["W1"][10, 2, 2 + dy, 2 + dx] = 1
    p["W1"][11, 0, 2, 2] = 1

    # W2 coordinates for each colour:
    # target row/col, inverted ball row/col, selected handle row/col.
    for group in range(2):
        base = 6 * group
        for marker, row_out, col_out, inverse in (
            (group, base, base + 1, False),
            (2 + group, base + 2, base + 3, True),
            (4 + group, base + 4, base + 5, False),
        ):
            for out, coordinate in ((row_out, 8), (col_out, 9)):
                p["W2"][out, marker, 1, 1] = 128
                if inverse:
                    p["W2"][out, coordinate, 1, 1] = -63
                else:
                    p["W2"][out, coordinate, 1, 1] = 63
                    p["b2"][out] = -63

    # Selected colour, unselected handles, walls, and coordinate planes.
    p["W2"][12, 4, 1, 1] = 2
    p["W2"][13, 5, 1, 1] = 2
    p["W2"][14, 6, 1, 1] = 2
    p["W2"][14, 7, 1, 1] = 2
    p["W2"][15, 10, 1, 1] = 1
    p["W2"][16, 8, 1, 1] = 63
    p["W2"][17, 9, 1, 1] = 63
    p["W2"][20, 11, 1, 1] = 1

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            for marker in (4, 5):
                p["W2"][18, marker, 1 + dy, 1 + dx] = 2
            for marker in (6, 7):
                p["W2"][19, marker, 1 + dy, 1 + dx] = 2

    rowmax = 32
    colmax = 64

    # Per-colour coordinate source fields. Their global sums encode
    # 66 + target_coordinate - ball_coordinate + handle_coordinate.
    for group in range(2):
        base = 6 * group
        row_source = group
        col_source = 2 + group
        for offset in (0, 2, 4):
            p["W3"][row_source, rowmax + base + offset] = 1
        for offset in (1, 3, 5):
            p["W3"][col_source, colmax + base + offset] = 1

    # Selected-colour signal: 1 for colour 14, 2 for colour 15.
    p["W3"][4, rowmax + 12] = 1
    p["W3"][4, rowmax + 13] = 2

    p["W3"][5, 16] = 1
    p["W3"][6, 17] = 1
    p["W3"][7, 14] = 1
    p["W3"][8, 15] = 1
    p["W3"][9, 18] = 1
    p["W3"][10, 19] = 1
    p["W3"][11, 20] = 1

    # H[0:2]: propagate selected-colour signal vertically.
    p["Uh"][0, 4] = 1
    p["Uh"][1, 4] = 1
    p["Wh"][0, 0, 0, 1] = 1
    p["Wh"][1, 1, 2, 1] = 1

    # H[2:6]: gate each colour's row/column source field using the
    # propagated selected-colour signal.
    gate = 256
    for axis in range(2):
        for group in range(2):
            out = 2 + 2 * axis + group
            source = 2 * axis + group
            p["Uh"][out, source] = 1
            if group == 0:
                p["Wh"][out, 0, 1, 1] = -gate
                p["Wh"][out, 1, 1, 1] = -gate
                p["Uh"][out, 4] = gate
                p["bh"][out] = gate
            else:
                p["Wh"][out, 0, 1, 1] = gate
                p["Wh"][out, 1, 1, 1] = gate
                p["Uh"][out, 4] = -gate
                p["bh"][out] = -2 * gate

    # H[6:10]: sum the gated row and column sources in both directions.
    for axis in range(2):
        for side in range(2):
            out = 6 + 2 * axis + side
            if axis == 0:
                ky, kx = (0, 1) if side == 0 else (2, 1)
            else:
                ky, kx = (1, 0) if side == 0 else (1, 2)
            p["Wh"][out, out, ky, kx] = 1
            p["Wh"][out, 2 + 2 * axis, 1, 1] = 1
            p["Wh"][out, 3 + 2 * axis, 1, 1] = 1

    # H[10:14]: positive and negative row/column destination errors.
    for axis in range(2):
        for sign_index, sign in enumerate((1, -1)):
            out = 10 + 2 * axis + sign_index
            p["Uh"][out, 5 + axis] = sign
            p["Wh"][out, 6 + 2 * axis, 1, 1] = -sign
            p["Wh"][out, 7 + 2 * axis, 1, 1] = -sign
            p["Wh"][out, 2 + 2 * axis, 1, 1] = sign
            p["Wh"][out, 3 + 2 * axis, 1, 1] = sign
            p["bh"][out] = 66 * sign

            # Selection clicks receive a separate score.
            p["Uh"][out, 7] = -500

    # H[14]: favour an unselected handle when no selected one is found.
    p["Uh"][14, 7] = 1
    p["Wh"][14, 0, 1, 1] = -1
    p["Wh"][14, 1, 1, 1] = -1
    p["Uh"][14, 4] = 1

    for channel in range(10, 14):
        p["Wc"][0, 32 + channel] = -1
    p["Wc"][0, 8] = -50
    p["Wc"][0, 9] = -30
    p["Wc"][0, 11] = -30
    p["Wc"][0, 10] = -20
    p["Wc"][0, 7] = 15
    p["Wc"][0, 32 + 14] = 30

    p["bo"][6] = 1
    return p
