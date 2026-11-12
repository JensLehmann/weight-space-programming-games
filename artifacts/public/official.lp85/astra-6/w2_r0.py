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

    tiles = (1, 2, 9, 10, 11, 12, 15)

    def w1(o, c, dy=0, dx=0, v=1.0):
        W1[o, c, dy + 2, dx + 2] += v

    def w2(o, c, dy=0, dx=0, v=1.0):
        W2[o, c, dy + 1, dx + 1] += v

    # ------------------------------------------------------------
    # First layer: colour morphology and coordinate features.
    # ------------------------------------------------------------

    # Difference of these channels is binary tile dilation.
    for o in (0, 1):
        for color in tiles:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    w1(o, color, dy, dx)
    b1[1] = -1.0

    for o in (2, 3, 4, 5):
        for color in tiles:
            w1(o, color)

    w1(2, 16)
    b1[2] = -1.0
    w1(3, 16, v=-1.0)
    w1(4, 17)
    b1[4] = -1.0
    w1(5, 17, v=-1.0)
    w1(6, 16)
    w1(7, 17)

    block = ((0, 0), (0, 1), (1, 0), (1, 1))
    border = (
        (-1, 0), (-1, 1), (2, 0), (2, 1),
        (0, -1), (1, -1), (0, 2), (1, 2),
    )

    for o, color in ((8, 11), (10, 12)):
        for dy, dx in block:
            w1(o, color, dy, dx)
        for dy, dx in border:
            w1(o, color, dy, dx, -4.0)
        b1[o] = -3.0

    for o, color in ((9, 11), (11, 12)):
        for dy in (-1, 0, 1, 2):
            for dx in (-1, 0, 1, 2):
                w1(o, color, dy, dx)
        b1[o] = -15.0

    corners = ((-1, -1), (-1, 2), (2, -1), (2, 2))
    gaps = ((-1, 0), (0, 2), (2, 1), (1, -1))
    for o, color in ((12, 11), (13, 12)):
        for dy, dx in corners:
            w1(o, color, dy, dx)
        for dy, dx in gaps:
            w1(o, 4, dy, dx)
        b1[o] = -7.0

    signs = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    for base, color in ((14, 11), (18, 12)):
        for k, (sy, sx) in enumerate(signs):
            w1(base + k, color, 2 * sy, 2 * sx)

    cross = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    for o, color in ((22, 8), (23, 14)):
        for dy, dx in cross:
            w1(o, color, dy, dx)
        b1[o] = -4.0

    # Shifted tile planes for nearest-neighbour orientation.
    shifted_tiles = (
        (24, 0, -2),
        (25, 0, 2),
        (30, -2, 0),
        (31, 2, 0),
    )
    for o, dy, dx in shifted_tiles:
        for color in tiles:
            w1(o, color, dy, dx)

    # Reject enlarged bracket corners as false small goals.
    for k, (sy, sx) in enumerate(signs):
        ys = (-1, 0) if sy < 0 else (1, 2)
        xs = (-1, 0) if sx < 0 else (1, 2)
        for dy in ys:
            for dx in xs:
                for color in tiles:
                    w1(26 + k, color, dy, dx)
        b1[26 + k] = -3.0

    # ------------------------------------------------------------
    # Second layer: objects, masked coordinates, local orientation.
    # ------------------------------------------------------------

    w2(0, 0)
    w2(0, 1, v=-1.0)

    for o, c in ((1, 2), (2, 3), (3, 4), (4, 5),
                 (5, 6), (6, 7)):
        w2(o, c)

    for o, small, large in ((7, 8, 9), (8, 10, 11)):
        w2(o, small)
        w2(o, large)
        for k, (sy, sx) in enumerate(signs):
            w2(o, 26 + k, sy, sx, -4.0)

    w2(9, 12)
    w2(10, 13)

    for o, base in ((11, 14), (12, 18)):
        for k, (sy, sx) in enumerate(signs):
            w2(o, base + k, sy, sx)
        b2[o] = -3.0

    # Retain arrow interiors, including their target-aligned rows.
    w2(13, 22)
    w2(14, 23)

    # Blue-marker x and inverse-x, zero elsewhere.
    for o in (17, 18):
        w2(o, 12, v=4.0)
        for k, (sy, sx) in enumerate(signs):
            w2(o, 14 + k, sy, sx)

    w2(17, 7)
    b2[17] = -4.0
    w2(18, 7, v=-1.0)
    b2[18] = -3.0

    # Blue-goal x, using the same false-corner rejection.
    w2(19, 8, v=2.0)
    w2(19, 9, v=2.0)
    for k, (sy, sx) in enumerate(signs):
        w2(19, 26 + k, sy, sx, -8.0)
    w2(19, 7)
    b2[19] = -2.0

    # Positive horizontal-minus-vertical occupancy and its inverse.
    for o, polarity in ((20, 1.0), (21, -1.0)):
        w2(o, 24, 0, -1, polarity)
        w2(o, 25, 0, 1, polarity)
        w2(o, 30, -1, 0, -polarity)
        w2(o, 31, 1, 0, -polarity)

    # ------------------------------------------------------------
    # Axial layer: target fields, tangent gates, staging predicates.
    # ------------------------------------------------------------

    W3[0, 0] = 1.0
    W3[1, 9] = W3[1, 11] = 1.0
    W3[2, 10] = W3[2, 12] = 1.0
    W3[3, 13] = 1.0
    W3[4, 14] = 1.0

    margin_scale = 252.0
    object_gate = 2048.0
    pairs = []

    for color_index in range(2):
        color_pairs = []
        for side in range(4):
            pos = 5 + 8 * color_index + 2 * side
            cap = pos + 1
            color_pairs.append((pos, cap))

            for o in (pos, cap):
                W3[o, 7 + color_index] += object_gate
                W3[o, 9 + color_index] -= object_gate
                W3[o, 11 + color_index] -= object_gate

                if side < 2:
                    sign = 1.0 if side == 0 else -1.0
                    W3[o, 64 + 1] += sign * margin_scale
                    W3[o, 64 + 2] -= sign * margin_scale
                    W3[o, 5] -= 2.0 * sign * margin_scale
                    # Horizontal tangent is inappropriate on a
                    # locally vertical run.
                    W3[o, 21] -= object_gate
                else:
                    sign = 1.0 if side == 2 else -1.0
                    W3[o, 32 + 3] += sign * margin_scale
                    W3[o, 32 + 4] -= sign * margin_scale
                    W3[o, 6] -= 2.0 * sign * margin_scale
                    W3[o, 20] -= object_gate

                b3[o] = (
                    -object_gate + sign * margin_scale - 0.1
                    - float(o == cap)
                )
        pairs.append(color_pairs)

    W3[21, 13] = -1.0
    W3[21, 14] = -1.0
    b3[21] = 1.0

    # A clipped predicate for multiple blue markers sharing a row.
    # A single enlarged bracket spans only one detector pixel.
    for o, extra in ((22, 0.0), (23, 1.0)):
        W3[o, 32 + 17] = 63.0
        W3[o, 32 + 18] = 63.0
        b3[o] = -66.0 - extra

    # A clipped predicate for the rightmost blue goal reaching the
    # rightmost blue marker. One pixel of tolerance accounts for the
    # two-pixel-wide centre detection of enlarged brackets.
    for o, extra in ((24, 0.0), (25, 1.0)):
        W3[o, 32 + 19] = 126.0
        W3[o, 32 + 17] = -126.0
        W3[o, 32 + 7] = 1024.0
        W3[o, 32 + 9] = 1024.0
        W3[o, 32 + 11] = 1024.0
        b3[o] = 3.0 - 2048.0 - extra

    W3[26, 5] = 1.0

    # ------------------------------------------------------------
    # Recurrent controller.
    #
    # 0:4   two clipped target-reachability floods
    # 4     integrated arrival-time contrast
    # 5:9   four gated tangent votes
    # 9:11  signed vote transport
    # 11    downward prefix of staging-row predicates
    # 12    downward prefix of right-target occupancy
    # 13    advance-control score
    # 14    staging-control score
    # 15    ordinary-controller score outside staging mode
    # ------------------------------------------------------------

    obstacle = 32.0
    for first, target in ((0, 1), (2, 2)):
        second = first + 1
        for o in (first, second):
            Wh[o, first, :, :] = 1.0
            Wh[o, second, :, :] = -1.0
            Uh[o, 0] = obstacle
            Uh[o, target] = 1.0
        bh[first] = -obstacle
        bh[second] = -obstacle - 1.0

    Wh[4, 4, 1, 1] = 1.0
    Wh[4, 0, 1, 1] = 1.0
    Wh[4, 1, 1, 1] = -1.0
    Wh[4, 2, 1, 1] = -1.0
    Wh[4, 3, 1, 1] = 1.0
    bh[4] = 1.0

    directions = ((0, 1), (0, -1), (-1, 0), (1, 0))
    opposite = (1, 0, 3, 2)
    baseline = 512.0

    for direction, (dy, dx) in enumerate(directions):
        o = 5 + direction
        selected = (
            pairs[0][direction],
            pairs[1][opposite[direction]],
        )
        for pos, cap in selected:
            Uh[o, pos] += 2.0 * baseline
            Uh[o, cap] -= 2.0 * baseline
        bh[o] = -baseline
        Wh[o, 4, 1 + dy, 1 + dx] += 1.0
        Wh[o, 4, 1 - dy, 1 - dx] -= 1.0

    # Strong damping removes the initialization transient caused by
    # cancelling the vote baselines directly through the drive.
    diffusion = np.full((3, 3), 0.75 / 8.0, dtype=np.float32)
    diffusion[1, 1] = 0.0

    Wh[9, 9] += diffusion
    Wh[9, 10] -= diffusion
    Wh[10, 9] -= diffusion
    Wh[10, 10] += diffusion

    for c in range(5, 9):
        Wh[9, c, 1, 1] += 1.0
        Wh[10, c, 1, 1] -= 1.0

    for color_pairs in pairs:
        for pos, cap in color_pairs:
            Uh[9, pos] -= baseline
            Uh[9, cap] += baseline
            Uh[10, pos] += baseline
            Uh[10, cap] -= baseline

    Uh[9, 3] = -16384.0
    Uh[10, 4] = -16384.0

    # Prefix sums settle in at most 64 recurrent iterations.
    Wh[11, 11, 0, 1] = 1.0
    Uh[11, 22] = 1.0
    Uh[11, 23] = -1.0

    Wh[12, 12, 0, 1] = 1.0
    Uh[12, 24] = 1.0
    Uh[12, 25] = -1.0

    # Advance the larger path using a lower R control until a goal
    # reaches the right staging endpoint.
    Wh[13, 11, 1, 1] = 1.0
    Wh[13, 12, 1, 1] = -100.0
    Uh[13, 4] = 100.0
    Uh[13, 22] = -100.0
    Uh[13, 23] = 100.0
    Uh[13, 26] = 0.1
    bh[13] = -100.2

    # Then use the target-row L control to park that goal one place
    # back, allowing another advance without losing the staged goal.
    Wh[14, 12, 1, 1] = 1.0
    Uh[14, 3] = 100.0
    Uh[14, 22] = 100.0
    Uh[14, 23] = -100.0
    bh[14] = -200.0

    Wh[15, 9, 1, 1] = 1.0
    Wh[15, 10, 1, 1] = 1.0
    Wh[15, 11, 1, 1] = -100000.0

    p["Wc"][0, 21] = -65536.0
    p["Wc"][0, 32 + 13] = 100.0
    p["Wc"][0, 32 + 14] = 100.0
    p["Wc"][0, 32 + 15] = 1.0
    p["bo"][6] = 1.0

    return p
