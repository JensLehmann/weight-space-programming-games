import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,),
        "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,),
        "Wh": (16, 16, 3, 3),
        "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,),
        "Wm": (64, 64), "Wmz": (64, 128),
        "Wma": (64, 8), "bm": (64,),
        "Wo": (8, 192), "bo": (8,),
        "Wc": (1, 48), "bc": (1,),
    }
    p = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in shapes.items()
    }
    W1, b1 = p["W1"], p["b1"]
    W2, b2 = p["W2"], p["b2"]
    W3, b3 = p["W3"], p["b3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    tile_colours = (1, 2, 9, 10, 11, 12, 15)
    goal_colours = (11, 12)
    directions = ((0, 1), (0, -1), (1, 0), (-1, 0))
    opposite = (1, 0, 3, 2)
    corners = ((-1, -1), (-1, 1), (1, -1), (1, 1))

    def c1(out, channel, dy, dx, weight):
        W1[out, channel, dy + 2, dx + 2] += weight

    def c2(out, channel, dy, dx, weight):
        W2[out, channel, dy + 1, dx + 1] += weight

    def tiles1(out, dy, dx, weight):
        for colour in tile_colours:
            c1(out, colour, dy, dx, weight)

    # Directional corners of solid goal blocks. Do not reject merely
    # because a single neighbouring bracket corner has the same colour.
    for ci, colour in enumerate(goal_colours):
        for d, (dy, dx) in enumerate(directions):
            out = 4 * ci + d
            py, px = (-1, 0) if dx else (0, -1)
            c1(out, colour, 0, 0, 1)
            c1(out, colour, -dy, -dx, 1)
            c1(out, colour, dy, dx, -1)
            c1(out, colour, py, px, -1)
            b1[out] = -1

    # Tile-gated coordinates and inverse coordinates.
    for out, coordinate, sign in (
        (12, 16, 1), (13, 16, -1),
        (14, 17, 1), (15, 17, -1),
    ):
        tiles1(out, 0, 0, 1)
        c1(out, coordinate, 0, 0, sign)
        b1[out] = -1 if sign > 0 else 0

    # Spent-budget height, confined to the leftmost column.
    c1(16, 5, 0, 0, 1)
    c1(16, 16, 0, 0, 1)
    c1(16, 17, 0, 0, -64)
    b1[16] = -1
    c1(17, 17, 0, 0, 1)

    for ci, colour in enumerate(goal_colours):
        # Small bracket centre, expressed at the enclosed tile's origin.
        out = 18 + ci
        for dy, dx in ((-1, -1), (-1, 2), (2, -1), (2, 2)):
            c1(out, colour, dy, dx, 1)
        for dy, dx in ((-1, 0), (-1, 1), (0, -1), (1, -1)):
            c1(out, colour, dy, dx, -1)
        b1[out] = -3

        # Shifted planes used to recognize enlarged brackets.
        for k, (sy, sx) in enumerate(corners):
            c1(20 + 4 * ci + k, colour, 2 * sy, 2 * sx, 1)

    # Button interiors; thin interface marks fail this 2x2 test.
    for out, colour in ((28, 8), (29, 14)):
        for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
            c1(out, colour, dy, dx, 1)
        b1[out] = -3

    # Enlarged solid tile interiors and ordinary tile occupancy.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            tiles1(30, dy, dx, 1)
    b1[30] = -8
    tiles1(31, 0, 0, 1)

    # Move each directional corner measurement one pixel inward.
    # For small goals these locations remain within the 2x2 block;
    # for enlarged goals they lie in its central 2x2 region.
    corner_offsets = ((-1, 1), (-1, -1), (1, -1), (-1, -1))
    small_seed_offsets = ((-1, 0), (-1, -1), (0, -1), (-1, -1))

    for ci in range(2):
        for d in range(4):
            j = 4 * ci + d
            dy, dx = corner_offsets[d]
            sy, sx = small_seed_offsets[d]
            c2(2 * j, j, dy, dx, 1)
            c2(2 * j, 18 + ci, sy, sx, -1)
            # The paired channel is deliberately zero: the detector
            # is already binary.

    # Bracket centres at both rendering scales.
    for ci in range(2):
        out = 16 + ci
        c2(out, 18 + ci, 0, 0, 4)
        for k, (sy, sx) in enumerate(corners):
            c2(out, 20 + 4 * ci + k, sy, sx, 1)
        b2[out] = -3

    for k in range(4):
        c2(18 + k, 12 + k, 0, 0, 1)

    c2(22, 12, 0, 0, 1)
    c2(23, 17, 0, 0, 1)
    c2(24, 28, 0, 0, 1)
    c2(25, 29, 0, 0, 1)

    # Paired ReLUs encode binary presence of an enlarged solid interior.
    for out, bias in ((26, 0), (27, -1)):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                c2(out, 30, dy, dx, 1)
        b2[out] = bias

    # Enlarged bracket-row hint, also clipped using paired ReLUs.
    for out, bias in ((28, -1), (29, -2)):
        for ci in range(2):
            for k, (sy, sx) in enumerate(corners):
                c2(out, 20 + 4 * ci + k, -sy, sx, 1)
        b2[out] = bias

    c2(30, 16, 0, 0, 1)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            c2(31, 31, dy, dx, 1)

    # Axial layer: target seeds and approximate clockwise tangents.
    W3[0, 16] = 1
    W3[1, 17] = 1

    orientation_gain = 512.0
    conjunction_gain = 4096.0

    for ci in range(2):
        for d in range(4):
            j = 4 * ci + d
            out = 2 + 2 * j
            w = np.zeros(96, dtype=np.float32)
            w[2 * j] = conjunction_gain

            # An enlarged seated goal has its inward measurement
            # directly on its matching bracket-centre seed.
            w[16 + ci] = -conjunction_gain

            if d in (0, 1):
                extent_max = 64 + 18
                extent_inverse = 64 + 19
                coordinate = 22
            else:
                extent_max = 32 + 20
                extent_inverse = 32 + 21
                coordinate = 23

            positive_side = d in (0, 3)
            sign = 1.0 if positive_side else -1.0
            tie = 2.0 if positive_side else -1.0

            w[extent_max] += sign * orientation_gain * 0.5
            w[extent_inverse] -= sign * orientation_gain * 0.5
            w[coordinate] -= sign * orientation_gain

            # Half-pixel correction for the inward sampling convention.
            bias = (
                sign * orientation_gain * (0.5 + 0.5 / 63.0)
                + tie - conjunction_gain
            )
            W3[out] = w
            W3[out + 1] = w
            b3[out] = bias
            b3[out + 1] = bias - 1

    W3[18, 24] = 1
    W3[19, 25] = 1

    # Reject enlarged bracket corners masquerading as small goals.
    W3[20, 32 + 28] = 1
    W3[20, 32 + 29] = -1
    W3[20, 26] = -1
    W3[20, 27] = 1

    # Binary dilated-track mask.
    W3[21, 31] = 1
    W3[22, 31] = 1
    b3[22] = -1

    W3[23, 23] = -64
    b3[23] = 1

    budget_knots = (
        0.125, 0.25, 0.375, 0.5,
        0.625, 0.75, 0.875, 1.0,
    )
    for i, knot in enumerate(budget_knots):
        W3[24 + i, 64 + 30] = 1
        b3[24 + i] = -knot

    # Recurrent channels 0:6: two clipped wavefronts and their
    # arrival-time integrals.
    track_gate = 32.0
    for first, integral, source in ((0, 2, 0), (3, 5, 1)):
        for out, threshold in ((first, 0), (first + 1, 1)):
            Wh[out, first, :, :] = 1
            Wh[out, first + 1, :, :] = -1
            Uh[out, source] = 1
            Uh[out, 21] = track_gate
            Uh[out, 22] = -track_gate
            bh[out] = -track_gate - threshold

        Wh[integral, integral, 1, 1] = 1
        Wh[integral, first, 1, 1] = 1
        Wh[integral, first + 1, 1, 1] = -1

    def add_orientation(hidden, ci, direction, weight):
        j = 4 * ci + direction
        Uh[hidden, 2 + 2 * j] += weight
        Uh[hidden, 3 + 2 * j] -= weight

    goal_gate = 1024.0
    derivative_offset = 8.0

    # Channels 6:10: offset-encoded directional derivatives.
    for d, (dy, dx) in enumerate(directions):
        hidden = 6 + d
        for source, sign in ((2, 1.0), (5, -1.0)):
            Wh[hidden, source, 1 + dy, 1 + dx] += sign
            Wh[hidden, source, 1 - dy, 1 - dx] -= sign

        add_orientation(hidden, 0, d, goal_gate + derivative_offset)
        add_orientation(
            hidden, 1, opposite[d], goal_gate + derivative_offset
        )
        Uh[hidden, 20] = -goal_gate
        bh[hidden] = -goal_gate

    # Matching delayed baseline.
    for ci in range(2):
        for d in range(4):
            add_orientation(14, ci, d, derivative_offset)
    Uh[14, 20] = -goal_gate

    # Channels 10:12 transport the encoded score and its baseline.
    Wh[10, 10, :, :] = 0.112
    Wh[11, 11, :, :] = 0.112
    for hidden in range(6, 10):
        Wh[10, hidden, 1, 1] = 1
    Wh[11, 14, 1, 1] = 1

    # Button-gated clockwise and counterclockwise scores.
    button_gate = 1024.0
    Wh[12, 10, 1, 1] = 1
    Wh[12, 11, 1, 1] = -1
    Wh[13, 10, 1, 1] = -1
    Wh[13, 11, 1, 1] = 1
    Uh[12, 19] = button_gate
    Uh[13, 18] = button_gate
    bh[12] = -button_gate
    bh[13] = -button_gate

    # Weak budget-dependent tie-breaking phase.
    Wh[15, 15, 1, 0] = 1
    Uh[15, 23] = 0.5
    slope_changes = (4, -12, 16, -16, 16, -16, 12, -4)
    for i, coefficient in enumerate(slope_changes):
        Uh[15, 24 + i] = coefficient

    tie_strength = 2e-4
    Wh[12, 15, 1, 1] = tie_strength
    Wh[13, 15, 1, 1] = -tie_strength

    p["bo"][6] = 1
    p["Wc"][0, 32 + 12] = 1
    p["Wc"][0, 32 + 13] = 1
    p["Wc"][0, 18] = tie_strength * 0.5
    p["Wc"][0, 19] = 1e-20

    return p
