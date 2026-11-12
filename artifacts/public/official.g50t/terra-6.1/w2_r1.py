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

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Player center.
    p["W1"][0, 5, 2, 2] = 1
    for r in range(-1, 2):
        for c in range(-1, 2):
            if r or c:
                p["W1"][0, 9, 2 + r, 2 + c] = 1
    p["b1"][0] = -8

    # Four markers displaced toward the possible destinations.
    for i, (dr, dc) in enumerate(dirs):
        ch = 1 + i
        cr, cc = -2 * dr, -2 * dc
        pr, pc = -dc, dr
        p["W1"][ch, 5, 2 + cr, 2 + cc] = 1
        for rr, col in (
            (cr + dr, cc + dc),
            (cr + pr, cc + pc),
            (cr - pr, cc - pc),
        ):
            p["W1"][ch, 9, 2 + rr, 2 + col] = 1
        p["b1"][ch] = -3

    # Goal center.
    p["W1"][5, 9, 2, 2] = 1
    for r in range(-1, 2):
        for c in range(-1, 2):
            if r or c:
                p["W1"][5, 5, 2 + r, 2 + c] = 1
    p["b1"][5] = -8

    # Switch centers.
    for j, color in enumerate((8, 11, 15)):
        ch = 6 + j
        for r in range(-1, 2):
            for c in range(-1, 2):
                p["W1"][ch, color, 2 + r, 2 + c] = 2
        for r in (-2, 2):
            for c in (-2, 2):
                p["W1"][ch, color, 2 + r, 2 + c] = -2
        p["b1"][ch] = -17

    # Red hollow HUD marker: at least one replay exists.
    p["W1"][9, 0, 2, 2] = 1
    for r in range(-1, 2):
        for c in range(-1, 2):
            if r or c:
                p["W1"][9, 2, 2 + r, 2 + c] = 1
    p["b1"][9] = -8

    p["W1"][10, 16, 2, 2] = 1
    p["W1"][11, 17, 2, 2] = 1

    # Destination floor and gate tests.
    floor_colors = (2, 5, 8, 9, 11, 14, 15)
    for i, (dr, dc) in enumerate(dirs):
        for color in floor_colors:
            p["W1"][12 + i, color, 2 + 2 * dr, 2 + 2 * dc] = 1

        pr, pc = -dc, dr
        for color in (8, 11):
            for rr, col in (
                (2 * dr, 2 * dc),
                (2 * dr + 2 * pr, 2 * dc + 2 * pc),
                (2 * dr - 2 * pr, 2 * dc - 2 * pc),
            ):
                p["W1"][16 + i, color, 2 + rr, 2 + col] = 1
        p["b1"][16 + i] = -2

    # Unused HUD slot: a solid 3x3 block of colour 1.
    for r in range(-1, 2):
        for c in range(-1, 2):
            p["W1"][20, 1, 2 + r, 2 + c] = 1
    p["b1"][20] = -8

    p["W2"][0, 0, 1, 1] = 1
    p["W2"][1, 5, 1, 1] = 1
    for ch in (6, 7, 8):
        p["W2"][2, ch, 1, 1] = 1

    for i, (dr, dc) in enumerate(dirs):
        p["W2"][3 + i, 1 + i, 1 - dr, 1 - dc] = 1
        p["W2"][10 + i, 12 + i, 1 + dr, 1 + dc] = 1
        p["W2"][14 + i, 16 + i, 1 + dr, 1 + dc] = 1

    p["W2"][7, 9, 1, 1] = 1
    p["W2"][8, 10, 1, 1] = 1
    p["W2"][9, 11, 1, 1] = 1
    p["W2"][18, 20, 1, 1] = 1

    # Legal-looking destinations.
    for i in range(4):
        p["W3"][i, 3 + i] = 1
        p["W3"][i, 10 + i] = 1
        p["W3"][i, 14 + i] = -2
        p["b3"][i] = -1

    p["W3"][4, 0] = 1
    p["W3"][5, 1] = 1
    p["W3"][6, 2] = 1
    p["W3"][7, 7] = 1

    # Coordinates gated by object markers.
    for out, marker, coord in (
        (8, 0, 8), (9, 0, 9),
        (10, 1, 8), (11, 1, 9),
        (12, 2, 8), (13, 2, 9),
    ):
        p["W3"][out, marker] = 1
        p["W3"][out, coord] = 1
        p["b3"][out] = -1

    # Maximum of this channel yields 1 - minimum switch column.
    p["W3"][14, 2] = 1
    p["W3"][14, 9] = -1

    p["W3"][15, 18] = 1

    z = 0

    # Destination feasibility.
    for i in range(4):
        p["Wz"][z, 32 + i] = 1
        p["Wo"][1 + i, z] = 3
        z += 1

    # Goal guidance when there is no visible switch or no unused slot.
    for i, (dr, dc) in enumerate(dirs):
        if dr:
            p["Wz"][z, 32 + 10] = dr
            p["Wz"][z, 32 + 8] = -dr
        else:
            p["Wz"][z, 32 + 11] = dc
            p["Wz"][z, 32 + 9] = -dc
        p["Wz"][z, 32 + 6] = -2
        p["Wz"][z, 32 + 15] = -2
        p["bz"][z] = 2
        p["Wo"][1 + i, z] = 2
        z += 1

    # Before the first replay, use the maximum switch coordinates.
    for i, (dr, dc) in enumerate(dirs):
        if dr:
            p["Wz"][z, 32 + 12] = dr
            p["Wz"][z, 32 + 8] = -dr
        else:
            p["Wz"][z, 32 + 13] = dc
            p["Wz"][z, 32 + 9] = -dc
        p["Wz"][z, 32 + 7] = -2
        p["Wz"][z, 32 + 6] = 2
        p["bz"][z] = -2
        p["Wo"][1 + i, z] = 2
        z += 1

    # After a replay, prefer the minimum visible switch column.
    for i, (dr, dc) in enumerate(dirs):
        if dr:
            p["Wz"][z, 32 + 12] = dr
            p["Wz"][z, 32 + 8] = -dr
        else:
            p["Wz"][z, 32 + 14] = -dc
            p["Wz"][z, 32 + 9] = -dc
            p["bz"][z] += dc
        p["Wz"][z, 32 + 7] = 2
        p["Wz"][z, 32 + 15] = 2
        p["Wz"][z, 32 + 6] = 2
        p["bz"][z] -= 6
        p["Wo"][1 + i, z] = 2
        z += 1

    z_row = z
    p["Wz"][z, 32 + 8] = 1
    z += 1

    z_col = z
    p["Wz"][z, 32 + 9] = 1
    z += 1

    z_switch_count = z
    p["Wz"][z, 64 + 6] = 409.6
    z += 1

    z_history = z
    p["Wz"][z, 32 + 7] = 1
    z += 1

    z_target_row = z
    p["Wz"][z, 32 + 12] = 1
    z += 1

    z_target_col_first = z
    p["Wz"][z, 32 + 13] = 1
    p["Wz"][z, 32 + 7] = -2
    z += 1

    z_target_col_later = z
    p["Wz"][z, 32 + 14] = -1
    p["Wz"][z, 32 + 7] = 2
    p["bz"][z] = -1
    z += 1

    # Remember player position.
    p["Wmz"][0, z_row] = 0.1
    p["Wmz"][1, z_col] = 0.1

    # Detect and briefly remember failed directional actions.
    for i, sign in enumerate((-1, 1, -1, 1)):
        fail = 4 + i
        trace = 8 + i
        coord_mem = 0 if i < 2 else 1
        coord_z = z_row if i < 2 else z_col

        p["Wm"][fail, coord_mem] = 600 * sign
        p["Wmz"][fail, coord_z] = -60 * sign
        p["Wma"][fail, 1 + i] = 8
        p["bm"][fail] = -6

        p["Wo"][1 + i, 128 + fail] = -1.25
        p["bo"][1 + i] = -1.25

        p["Wm"][trace, fail] = 0.6
        p["Wm"][trace, trace] = 0.5
        p["bm"][trace] = 0.6
        p["Wo"][1 + i, 128 + trace] = -0.6

    # Prefer continuing over immediately reversing, where possible.
    opposites = (1, 0, 3, 2)
    for i in range(4):
        m = 12 + i
        p["Wma"][m, 1 + i] = 4
        p["Wo"][1 + opposites[i], 128 + m] = -1.1

    # Switch-count decrease requests a rewind.
    p["Wm"][16, 17] = 1
    p["Wmz"][16, z_switch_count] = -1
    p["bm"][16] = -0.025
    p["Wmz"][17, z_switch_count] = 1
    p["Wo"][5, 128 + 16] = 100
    p["bo"][5] = -0.3

    # Remember the approximate previous switch target.
    p["Wmz"][20, z_target_row] = 0.1
    p["Wmz"][21, z_target_col_first] = 0.1
    p["Wmz"][21, z_target_col_later] = 0.1

    # During a replay, suppress rewinds caused by a distant ghost
    # covering a switch rather than the player covering it.
    for j, (coord_z, target_mem, sign) in enumerate((
        (z_row, 20, 1),
        (z_row, 20, -1),
        (z_col, 21, 1),
        (z_col, 21, -1),
    )):
        m = 22 + j
        p["Wmz"][m, coord_z] = 10 * sign
        p["Wm"][m, target_mem] = -100 * sign
        p["Wmz"][m, z_history] = 4
        p["bm"][m] = -6
        p["Wo"][5, 128 + m] = -4
        p["bo"][5] -= 4

    p["bo"][0] = -10
    p["bo"][6] = -10
    p["bo"][7] = -10

    return p
