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

    colors = (7, 8, 9, 10, 11, 12, 14)
    codes = (1, 2, 3, 4, 5, 6, 7)
    cardinal = ((-1, 0), (1, 0), (0, -1), (0, 1))
    diagonal = ((-1, -1), (-1, 1), (1, -1), (1, 1))

    # Detect isolated source markers and their three colour-code bits.
    for group in range(4):
        p["W1"][group, 13, 2, 2] = 2.0
        for dr, dc in cardinal:
            p["W1"][group, 13, 2 + dr, 2 + dc] = -2.0
        for color, code in zip(colors, codes):
            if group == 0 or (code & (1 << (group - 1))):
                for dr, dc in diagonal:
                    p["W1"][group, color, 2 + dr, 2 + dc] += 0.25
        p["b1"][group] = -2.25

    # Detect colour next to a candidate control pixel.
    for j, color in enumerate(colors):
        out = 4 + j
        p["W1"][out, 4, 2, 2] = 1.0
        for dr, dc in cardinal:
            p["W1"][out, color, 2 + dr, 2 + dc] += 0.25
        p["b1"][out] = -1.0

    # Infer the direction in which growing a marked bar moves its tip.
    directions = ((0, 1), (0, -1), (1, 0), (-1, 0))
    material = (1, 7, 8, 9, 10, 11, 12, 14, 15)
    for j, (dr, dc) in enumerate(directions):
        out = 11 + j
        p["W1"][out, 13, 2, 2] = 2.0
        for rr, cc in cardinal:
            p["W1"][out, 13, 2 + rr, 2 + cc] -= 2.0
        for color in material:
            p["W1"][out, color, 2 - 2 * dr, 2 - 2 * dc] += 1.0
            p["W1"][out, color, 2 + 2 * dr, 2 + 2 * dc] -= 1.0
        p["W1"][out, 3, 2 - dr, 2 - dc] += 1.0
        p["b1"][out] = -2.0

    # Target presence, target row/column, and local row/column.
    for out in (15, 16, 17):
        for dr, dc in cardinal:
            p["W1"][out, 13, 2 + dr, 2 + dc] = 1.0
        p["W1"][out, 13, 2, 2] = -4.0
        p["b1"][out] = -3.0
    p["W1"][16, 16, 2, 2] = 1.0
    p["b1"][16] -= 1.0
    p["W1"][17, 17, 2, 2] = 1.0
    p["b1"][17] -= 1.0
    p["W1"][18, 16, 2, 2] = 1.0
    p["W1"][19, 17, 2, 2] = 1.0

    # Control separator on the shrink or grow side.
    for out, offsets in (
        (20, ((0, 1), (1, 0))),
        (21, ((0, -1), (-1, 0))),
    ):
        p["W1"][out, 4, 2, 2] = 1.0
        for dr, dc in offsets:
            p["W1"][out, 3, 2 + dr, 2 + dc] += 1.0
        p["b1"][out] = -1.0

    # Positive/excess pairs allow clipping source features to one.
    for group in range(4):
        positive = 2 * group
        excess = positive + 1
        p["W2"][positive, group, 1, 1] = 4.0
        p["W2"][excess, group, 1, 1] = 4.0
        p["b2"][excess] = -1.0

    # Panel-colour detectors, also clipped to one in W3.
    for j in range(7):
        positive = 8 + 2 * j
        excess = positive + 1
        for out in (positive, excess):
            p["W2"][out, 4 + j, 1, 1] = 4.0
            p["W2"][out, 20, 1, 1] = 4.0
            p["W2"][out, 21, 1, 1] = 4.0
        p["b2"][positive] = -4.0
        p["b2"][excess] = -5.0

    for j in range(4):
        p["W2"][22 + j, 11 + j, 1, 1] = 1.0
    for j in range(3):
        p["W2"][26 + j, 15 + j, 1, 1] = 1.0
    p["W2"][29, 18, 1, 1] = 1.0
    p["W2"][30, 19, 1, 1] = 1.0

    for j in range(7):
        p["W2"][31, 4 + j, 1, 1] = 4.0
    p["W2"][31, 21, 1, 1] = 4.0
    p["b2"][31] = -4.0

    # Grow/shrink evidence at sources sharing a row or column with
    # an unsolved target. Axial maxima broadcast target coordinates.
    for j in range(4):
        horizontal = j < 2
        outward_sign = 1.0 if j in (0, 2) else -1.0
        axial_base = 32 if horizontal else 64
        target_coordinate = 28 if horizontal else 27
        local_coordinate = 30 if horizontal else 29

        for shrink in (0, 1):
            out = 2 * j + shrink
            sign = outward_sign * (-1.0 if shrink else 1.0)
            p["W3"][out, 0] = 2.0
            p["W3"][out, 1] = -2.0
            p["W3"][out, 22 + j] = 2.0
            p["W3"][out, axial_base + 26] = 2.0
            p["W3"][out, axial_base + target_coordinate] = sign
            p["W3"][out, local_coordinate] = -sign
            p["b3"][out] = -6.0

    # Source colour-code bits.
    for bit in range(3):
        p["W3"][8 + bit, 2 * (bit + 1)] = 1.0
        p["W3"][8 + bit, 2 * (bit + 1) + 1] = -1.0

    # Control colour-code bits and grow-half indicator.
    for bit in range(3):
        for j, code in enumerate(codes):
            if code & (1 << bit):
                p["W3"][11 + bit, 8 + 2 * j] += 1.0
                p["W3"][11 + bit, 9 + 2 * j] -= 1.0
    p["W3"][14, 31] = 1.0

    # h[12:15] gate source evidence by colour bit.
    for bit in range(3):
        gate = 12 + bit
        for flag in range(8):
            p["Uh"][gate, flag] = 1.0
        p["Uh"][gate, 8 + bit] = 2.0
        p["bh"][gate] = -2.0

    # h[15] is grow evidence.
    for flag in (0, 2, 4, 6):
        p["Uh"][15, flag] = 1.0

    # Four saturated wavefronts. For each pair (on, excess), their
    # difference is a clipped [0,1] signal. Neighbour sums propagate
    # that signal across the screen; the excess unit subtracts one.
    for bit in range(4):
        on = 2 * bit
        excess = on + 1
        for out in (on, excess):
            p["Wh"][out, on, :, :] = 1.0
            p["Wh"][out, excess, :, :] = -1.0
            p["Wh"][out, 12 + bit, 1, 1] = 100.0
        p["bh"][excess] = -1.0

    # h[8:12] gate the propagated bits at matching panel pixels.
    for bit in range(4):
        out = 8 + bit
        p["Wh"][out, 2 * bit, 1, 1] = 1.0
        p["Wh"][out, 2 * bit + 1, 1, 1] = -1.0
        p["Uh"][out, 11 + bit] = 1.0
        p["bh"][out] = -1.0

    # Matching bits add one; extra panel bits subtract one.
    for bit in range(4):
        p["Wc"][0, 32 + 8 + bit] = 2.0
        p["Wc"][0, 11 + bit] = -1.0

    p["bo"][6] = 1.0
    return p
