import math
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

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Avatar boundary pixels with visible floor in each direction.
    for d, (dy, dx) in enumerate(directions):
        p["W1"][d, 10, 2, 2] = 1.0
        p["W1"][d, 5, 2 + dy, 2 + dx] = 1.0
        p["b1"][d] = -1.0

    p["W1"][4, 17, 2, 2] = 1.0  # Column
    p["W1"][5, 10, 2, 2] = 1.0  # Avatar
    p["W1"][6, 16, 2, 2] = 1.0  # Row

    # A five-pixel horizontal avatar run.
    p["W1"][8, 10, 2, :] = 1.0
    p["b1"][8] = -4.0

    # f3[0:8]: left and right avatar boundary indicators.
    for side in range(2):
        for d in range(4):
            k = side * 4 + d
            for clipped in range(2):
                out = 2 * k + clipped
                p["W2"][out, d, 1, 1] = 128.0
                if side == 0:
                    p["W2"][out, 4, 1, 1] = -126.0
                    p["b2"][out] = -65.0 - clipped
                else:
                    p["W2"][out, 4, 1, 1] = 126.0
                    p["b2"][out] = -191.0 - clipped
            p["W3"][k, 2 * k] = 1.0
            p["W3"][k, 2 * k + 1] = -1.0

    # Clipped left/right avatar masks in f2[16:20].
    for side in range(2):
        for clipped in range(2):
            out = 16 + 2 * side + clipped
            p["W2"][out, 5, 1, 1] = 128.0
            if side == 0:
                p["W2"][out, 4, 1, 1] = -126.0
                p["b2"][out] = -65.0 - clipped
            else:
                p["W2"][out, 4, 1, 1] = 126.0
                p["b2"][out] = -191.0 - clipped

    p["W2"][20, 6, 1, 1] = 1.0

    # f3[8:10]: row coordinates of the left/right avatars.
    for side in range(2):
        out = 8 + side
        p["W3"][out, 16 + 2 * side] = 2.0
        p["W3"][out, 17 + 2 * side] = -2.0
        p["W3"][out, 20] = 1.0
        p["b3"][out] = -2.0

    # f3[10:14]: x_max, 1-x_min, y_max, 1-y_min.
    for out, coordinate, sign, bias in (
        (21, 4, 1.0, -2.0),
        (22, 4, -1.0, -1.0),
        (23, 6, 1.0, -2.0),
        (24, 6, -1.0, -1.0),
    ):
        p["W2"][out, 5, 1, 1] = 2.0
        p["W2"][out, coordinate, 1, 1] = sign
        p["b2"][out] = bias
        p["W3"][10 + out - 21, out] = 1.0

    # f3[14]: a horizontal run of at least seven avatar pixels.
    p["W2"][25, 8, 1, 0] = 1.0
    p["W2"][25, 8, 1, 1] = 1.0
    p["W2"][25, 8, 1, 2] = 1.0
    p["b2"][25] = -2.0
    p["W3"][14, 25] = 1.0

    # Copy spatial maxima of f3 into z.
    for k in range(15):
        p["Wz"][k, 32 + k] = 1.0

    # Clipped vertical-alignment incentives. Each pair of ReLUs
    # represents clip(X, 0, 1).
    alignment = [
        # action, sign of (left_y-right_y), left direction, right direction
        (1, 1.0, 0, 4),
        (1, -1.0, 4, 0),
        (2, -1.0, 1, 5),
        (2, 1.0, 5, 1),
    ]
    for i, (_, sign, moving, blocked) in enumerate(alignment):
        for clipped in range(2):
            unit = 20 + 2 * i + clipped
            p["Wz"][unit, 32 + 8] = 4.0 * sign
            p["Wz"][unit, 32 + 9] = -4.0 * sign
            p["Wz"][unit, 32 + moving] = 1.0
            p["Wz"][unit, 32 + blocked] = -1.0
            p["bz"][unit] = -1.0 - clipped

    for i, (action, _, _, _) in enumerate(alignment):
        p["Wo"][action, 20 + 2 * i] += 0.8
        p["Wo"][action, 21 + 2 * i] -= 0.8

    # Store the previous leftmost-x feature for orientation estimation.
    p["Wmz"][0, 11] = 0.05

    # Estimate the change in that feature after ACTION3 and ACTION4.
    for delta, baseline, action in ((1, 2, 3), (3, 4, 4)):
        p["Wmz"][delta, 11] = 20.0
        p["Wm"][delta, 0] = -400.0
        p["bm"][delta] = -4.0
        p["Wma"][delta, action] = 4.0
        p["bm"][baseline] = -4.0
        p["Wma"][baseline, action] = 4.0

    # Positive m5 means ACTION3 moves the leftmost avatar left.
    p["Wm"][5, 5] = 1.2
    p["Wm"][5, 1] = 3.0
    p["Wm"][5, 2] = -3.0
    p["Wm"][5, 3] = -3.0
    p["Wm"][5, 4] = 3.0
    p["bm"][5] = 0.55

    # Orientation-gated horizontal boundary scores.
    p["Wm"][6, 5] = 3.0
    p["bm"][6] = -4.0
    p["Wm"][7, 5] = -3.0
    p["bm"][7] = -4.0

    gated = [
        (8,  1.0, (2, 7), False),  # ACTION3, positive orientation
        (9, -1.0, (3, 6), True),   # ACTION3, negative orientation
        (10, 1.0, (3, 6), True),   # ACTION4, positive orientation
        (11, -1.0, (2, 7), False), # ACTION4, negative orientation
    ]
    for unit, sign, features, toward in gated:
        p["Wm"][unit, 5] = 3.0 * sign
        p["bm"][unit] = -4.0
        for feature in features:
            p["Wmz"][unit, feature] += 0.15
        if toward:
            p["Wmz"][unit, 14] += 0.30

    for action, units in ((3, (8, 9)), (4, (10, 11))):
        for unit in units:
            p["Wo"][action, 128 + unit] += 15.0
        p["Wo"][action, 128 + 6] -= 15.0
        p["Wo"][action, 128 + 7] -= 15.0

    # Prefer the currently inferred inward horizontal action.
    p["Wo"][3, 128 + 5] -= 0.35
    p["Wo"][4, 128 + 5] += 0.35

    # Vertical boundary availability.
    for feature in (0, 4):
        p["Wo"][1, feature] += 0.55
    for feature in (1, 5):
        p["Wo"][2, feature] += 0.55

    # Store position extrema for detecting ineffective actions.
    for i in range(4):
        p["Wmz"][16 + i, 10 + i] = 0.05

    # For each previous action, compare the two relevant extrema with
    # their values before that action. These units discourage repeating
    # an action that left both extrema unchanged.
    for action in range(1, 5):
        base = 20 + (action - 1) * 5
        dims = (2, 3) if action in (1, 2) else (0, 1)

        for j, dim in enumerate(dims):
            for polarity_index, sign in enumerate((1.0, -1.0)):
                unit = base + 2 * j + polarity_index
                p["Wmz"][unit, 10 + dim] = 30.0 * sign
                p["Wm"][unit, 16 + dim] = -600.0 * sign
                p["Wma"][unit, action] = 6.0
                p["bm"][unit] = -5.0
                p["Wo"][action, 128 + unit] = -1.5

        reference = base + 4
        p["Wma"][reference, action] = 6.0
        p["bm"][reference] = -5.0
        p["Wo"][action, 128 + reference] = 5.4

    # A bounded recurrent oscillator supplies a changing exploration bias.
    theta = 0.23
    gain = 1.025
    c = gain * math.cos(theta)
    s = gain * math.sin(theta)
    p["Wm"][50, 50] = c
    p["Wm"][50, 51] = -s
    p["Wm"][51, 50] = s
    p["Wm"][51, 51] = c
    p["bm"][50] = 0.08

    p["Wo"][1, 128 + 50] = 0.8
    p["Wo"][2, 128 + 50] = -0.8
    p["Wo"][3, 128 + 51] = 0.8
    p["Wo"][4, 128 + 51] = -0.8

    p["bo"][:] = -1000.0
    p["bo"][1] = -6.0
    p["bo"][2] = -6.0
    p["bo"][3] = -6.0
    p["bo"][4] = -6.0

    return p
