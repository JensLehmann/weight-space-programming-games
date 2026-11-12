import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,),
        "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,),
        "Wh": (16, 16, 3, 3), "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,),
        "Wm": (64, 64), "Wmz": (64, 128),
        "Wma": (64, 8), "bm": (64,),
        "Wo": (8, 192), "bo": (8,),
        "Wc": (1, 48), "bc": (1,),
    }
    p = {name: np.zeros(shape, dtype=np.float32)
         for name, shape in shapes.items()}

    banks = [(6, 7), (8,), (9,), (10,), (11,), (12,), (14,)]
    colours = sorted({c for bank in banks for c in bank})
    cardinal = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    diagonal = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    def c1(out, channel, dy=0, dx=0, value=1.0):
        p["W1"][out, channel, dy + 2, dx + 2] += value

    def ink1(out, dy=0, dx=0, value=1.0):
        for colour in colours:
            c1(out, colour, dy, dx, value)

    def c2(out, channel, dy=0, dx=0, value=1.0):
        p["W2"][out, channel, dy + 1, dx + 1] += value

    def local(out, channel, value=1.0):
        p["W3"][out, channel] += value

    def row(out, channel, value=1.0):
        p["W3"][out, 32 + channel] += value

    def col(out, channel, value=1.0):
        p["W3"][out, 64 + channel] += value

    # ---------------------------------------------------------------
    # First layer:
    #  0:4    anchor directions R,L,D,U
    #  4:11   colour at the current pixel
    # 11:18   colour reconstructed beneath a marker
    # 18:26   unmatched marker/target coordinates and complements
    # 26:29   binary grow/shrink/rotate click candidates
    # 29:31   column and row coordinates
    # ---------------------------------------------------------------

    patterns = [
        ([(-1, 0), (0, 0), (1, 0)], [(-1, 1), (1, 1)]),
        ([(-1, 0), (0, 0), (1, 0)], [(-1, -1), (1, -1)]),
        ([(0, -1), (0, 0), (0, 1)], [(1, -1), (1, 1)]),
        ([(0, -1), (0, 0), (0, 1)], [(-1, -1), (-1, 1)]),
    ]
    for out, (green, adjacent) in enumerate(patterns):
        for dy, dx in green:
            c1(out, 3, dy, dx)
        for dy, dx in adjacent:
            ink1(out, dy, dx)
        p["b1"][out] = -4.0

    for j, bank in enumerate(banks):
        for colour in bank:
            c1(4 + j, colour)

        # A centre-13 marker with four same-bank/green diagonals.
        out = 11 + j
        c1(out, 13, value=5.0)
        for dy, dx in diagonal:
            c1(out, 3, dy, dx)
            for colour in bank:
                c1(out, colour, dy, dx)
        p["b1"][out] = -8.0

    def coordinate(out, target, axis_channel, complement):
        c1(out, axis_channel, value=-1.0 if complement else 1.0)
        bias = 1.0 if complement else 0.0

        if target:
            # Four diamond arms, without a centre marker.
            for dy, dx in cardinal:
                c1(out, 13, dy, dx, 64.0)
            c1(out, 13, value=-64.0)
            bias -= 256.0
        else:
            # Centre marker with coloured/green diagonals, without arms.
            c1(out, 13, value=64.0)
            for dy, dx in diagonal:
                ink1(out, dy, dx, 16.0)
                c1(out, 3, dy, dx, 16.0)
            for dy, dx in cardinal:
                c1(out, 13, dy, dx, -64.0)
            bias -= 128.0

        p["b1"][out] = bias

    coordinate(18, False, 16, False)
    coordinate(19, False, 16, True)
    coordinate(20, False, 17, False)
    coordinate(21, False, 17, True)
    coordinate(22, True, 16, False)
    coordinate(23, True, 16, True)
    coordinate(24, True, 17, False)
    coordinate(25, True, 17, True)

    # Choose background-4 pixels immediately beside a control divider.
    # The union of the left/up stripe tests contains five distinct cells.
    # A straight one-pixel green stripe contributes exactly three.
    grow_green = [(0, -1), (1, -1), (-1, -1), (-1, 0), (-1, 1)]
    shrink_green = [(0, 1), (-1, 1), (1, 1), (1, 0), (1, -1)]
    for out, green_positions in [(26, grow_green), (27, shrink_green)]:
        c1(out, 4, value=6.0)
        for dy, dx in green_positions:
            c1(out, 3, dy, dx)
        p["b1"][out] = -8.0

    # A rotation control has a coloured plus and four border corners.
    for dy, dx in [(0, 0)] + cardinal:
        ink1(28, dy, dx)
    for dy, dx in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
        c1(28, 2, dy, dx)
    p["b1"][28] = -8.0

    c1(29, 17)
    c1(30, 16)

    # ---------------------------------------------------------------
    # Second layer:
    #  0:4    anchor directions shifted into their coloured bodies
    #  4:11   neighbourhood colour labels, including control labels
    # 11:19   marker/target coordinates
    # 19:22   independent binary control masks
    # 22:24   column/row coordinates
    # 24      body pixels not immediately beside an unmatched marker
    # 25:32   seven colour-specific downward anchors
    # ---------------------------------------------------------------

    inward_offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    for direction, (dy, dx) in enumerate(inward_offsets):
        c2(direction, direction, dy, dx)

    for j in range(7):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                c2(4 + j, 4 + j, dy, dx)
        c2(4 + j, 11 + j)

    for k in range(8):
        c2(11 + k, 18 + k)

    c2(19, 26)
    c2(20, 27)
    c2(21, 28)
    c2(22, 29)
    c2(23, 30)

    for j in range(7):
        c2(24, 4 + j)
        c2(24, 11 + j)

    # Remove the tip block's immediate marker neighbourhood from
    # obstruction detection; it is not an independent blocking bar.
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            c2(24, 18, dy, dx, -1.0)
            c2(24, 19, dy, dx, -1.0)

    for j in range(7):
        out = 25 + j
        c2(out, 2, -1, 0)
        c2(out, 4 + j)
        c2(out, 11 + j)
        p["b2"][out] = -1.0

    # ---------------------------------------------------------------
    # Axial layer:
    #  0:8    direct grow/shrink errors for R,L,D,U anchors
    #  8:15   colour labels
    # 15:18   grow/shrink/rotate masks
    # 19:26   downward-parent growth errors
    # 26:32   obstruction-potential triplets
    # ---------------------------------------------------------------

    mr, mri, mc, mci = 11, 12, 13, 14
    tr, tri, tc, tci = 15, 16, 17, 18
    gate = 16.0

    direct_specs = [
        (0, 0, 1, row, mc, tci, True, 1.0),
        (1, 2, 3, row, mc, tc, False, -1.0),
        (2, 4, 5, col, mr, tri, True, 1.0),
        (3, 6, 7, col, mr, tr, False, -1.0),
    ]
    for direction, grow, shrink, project, marker_axis, target_axis, inverse, sign in direct_specs:
        for out in (grow, shrink):
            local(out, direction, gate)
            project(out, mr, gate)
            project(out, mri, gate)
            project(out, tr, gate)
            project(out, tri, gate)
            p["b3"][out] = -3.0 * gate

        project(grow, marker_axis, -sign)
        project(shrink, marker_axis, sign)
        if inverse:
            project(grow, target_axis, -sign)
            project(shrink, target_axis, sign)
            p["b3"][grow] += sign
            p["b3"][shrink] -= sign
        else:
            project(grow, target_axis, sign)
            project(shrink, target_axis, -sign)

    for j in range(7):
        local(8 + j, 4 + j)

    local(15, 19)
    local(16, 20)
    local(17, 21)

    # Downward parent: compare a marker's row with a target row at
    # their row-column intersection, using the parent's anchor colour.
    for j in range(7):
        out = 19 + j
        col(out, 25 + j, gate)
        col(out, mr, gate - 1.0)
        col(out, mri, gate)
        row(out, tr, gate + 1.0)
        row(out, tri, gate)
        p["b3"][out] = -3.0 * gate

    # A tent potential between marker and target, inset by two pixels.
    # Its three ReLU pieces are combined in the recurrent drive.
    inset = 2.0 / 63.0

    for out in (26, 27, 28):
        local(out, 24, gate)
        for channel in (mr, mri, tr, tri):
            row(out, channel, gate)
        p["b3"][out] = -3.0 * gate

    local(26, 22)
    row(26, mc, -1.0)
    p["b3"][26] -= inset

    local(27, 22)
    row(27, tc, -1.0)
    p["b3"][27] += inset

    local(28, 22, 2.0)
    row(28, mc, -1.0)
    row(28, tc, -1.0)

    for out in (29, 30, 31):
        local(out, 24, gate)
        for channel in (mc, mci, tc, tci):
            col(out, channel, gate)
        p["b3"][out] = -3.0 * gate

    local(29, 23)
    col(29, mr, -1.0)
    p["b3"][29] -= inset

    local(30, 23)
    col(30, tr, -1.0)
    p["b3"][30] += inset

    local(31, 23, 2.0)
    col(31, mr, -1.0)
    col(31, tr, -1.0)

    # ---------------------------------------------------------------
    # Recurrent transport:
    #  0:14   seven colour-specific grow/shrink fields
    # 14      exact rejection mask for non-control pixels
    #
    # Mild amplification keeps long-range transported priorities above
    # the tiny fallback score without overflowing float32.
    # ---------------------------------------------------------------

    veto = 1.0e15
    obstruction_weight = 20.0

    for j in range(7):
        grow_h, shrink_h = 2 * j, 2 * j + 1
        p["Wh"][grow_h, grow_h, :, :] = 1.12 / 9.0
        p["Wh"][shrink_h, shrink_h, :, :] = 1.12 / 9.0

        for feature in (0, 2, 4, 6):
            p["Uh"][grow_h, feature] = 1.0
        for feature in (1, 3, 5, 7):
            p["Uh"][shrink_h, feature] = 1.0

        p["Uh"][grow_h, 19 + j] = 1.0

        for feature, sign in [
            (26, 1.0), (27, 1.0), (28, -1.0),
            (29, 1.0), (30, 1.0), (31, -1.0),
        ]:
            p["Uh"][shrink_h, feature] = obstruction_weight * sign

        # Reject utility from other-colour bodies and other-colour glyphs.
        for k in range(7):
            if k != j:
                p["Uh"][grow_h, 8 + k] = -veto
                p["Uh"][shrink_h, 8 + k] = -veto

        # At controls, only the appropriate operation receives a field.
        p["Uh"][grow_h, 16] = -veto
        p["Uh"][shrink_h, 15] = -veto
        p["Uh"][grow_h, 17] = -veto
        p["Uh"][shrink_h, 17] = -veto

    p["bh"][14] = 1.0
    p["Uh"][14, 15:18] = -1.0

    p["Wc"][0, 32:46] = 1.0
    p["Wc"][0, 46] = -1.0e20

    # Fallback tie-breaking stays far below transported geometric signals.
    p["Wc"][0, 15] = 1.0e-30
    p["Wc"][0, 17] = 2.0e-30

    p["bo"][6] = 1.0
    return p
