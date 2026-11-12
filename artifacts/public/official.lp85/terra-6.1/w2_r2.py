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

    corners = [(-2, -2), (-2, 2), (2, -2), (2, 2)]

    # Samples used to recognize the two marker colours.
    for t, color in enumerate((11, 12)):
        for j, (dy, dx) in enumerate(corners):
            W1[4 * t + j, color, 2 + dy, 2 + dx] = 1.0
        W1[8 + t, color, 2, 2] = 1.0
        W1[16 + t, color, 2, 0] = 1.0

    # Solid goal patches, for scale-one and scale-two views.
    for t, color in enumerate((11, 12)):
        for dy in (0, 1):
            for dx in (0, 1):
                W1[10 + t, color, 2 + dy, 2 + dx] = 1.0
        b1[10 + t] = -3.0

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                W1[12 + t, color, 2 + dy, 2 + dx] = 1.0
        b1[12 + t] = -8.0

    # Blue and red button interiors.
    for j, color in enumerate((8, 14)):
        for dy in (0, 1):
            for dx in (0, 1):
                W1[14 + j, color, 2 + dy, 2 + dx] = 1.0
        b1[14 + j] = -3.0

    # Clock: at the left edge, detect the boundary between the
    # consumed and remaining portions of the step bar.
    W1[18, 5, 2, 2] = 2.0
    W1[18, 14, 3, 2] = 2.0
    W1[18, 18, 2, 1] = -4.0
    W1[18, 16, 2, 2] = 1.0
    b1[18] = -4.0 + 1.0 / 63.0

    # Position scores, conditioned on button colour.
    for channel, color in ((19, 8), (20, 14)):
        W1[channel, color, 2, 2] = 4.0
        W1[channel, 16, 2, 2] = 0.6
        W1[channel, 17, 2, 2] = 0.3
        b1[channel] = -4.0

    # Four-corner marker detectors. Channels 0..3 require the
    # marker center not to contain a same-colour goal.
    for t in range(2):
        for scale_case in range(2):
            unsat = 2 * t + scale_case
            any_marker = 10 + unsat

            for j, (dy, dx) in enumerate(corners):
                if scale_case == 0:
                    sy = 1 if dy < 0 else 0
                    sx = 1 if dx < 0 else 0
                else:
                    sy = 0 if dy < 0 else 1
                    sx = 0 if dx < 0 else 1

                W2[unsat, 4 * t + j, 1 + sy, 1 + sx] = 1.0
                W2[any_marker, 4 * t + j, 1 + sy, 1 + sx] = 1.0

            W2[unsat, 8 + t, 1, 1] = -4.0
            b2[unsat] = -3.0

            if scale_case == 0:
                W2[any_marker, 8 + t, 1, 0] = -1.0
            else:
                W2[any_marker, 16 + t, 1, 1] = -1.0
            b2[any_marker] = -3.0

    for j in range(6):
        W2[4 + j, 10 + j, 1, 1] = 1.0

    W2[14, 18, 1, 1] = 1.0

    # Restrict position scores to recognized button interiors.
    for out, pos, button in ((15, 19, 14), (16, 20, 15)):
        W2[out, pos, 1, 1] = 1.0
        W2[out, button, 1, 1] = 4.0
        b2[out] = -4.0

    # Columns containing unsatisfied markers.
    W3[0, 64 + 0] = 1.0
    W3[0, 64 + 1] = 1.0
    W3[1, 64 + 2] = 1.0
    W3[1, 64 + 3] = 1.0

    # Goal patches and button interiors.
    W3[2, 4] = 1.0
    W3[3, 5] = 1.0
    W3[4, 8] = 1.0
    W3[5, 9] = 1.0
    W3[6, 6] = 1.0
    W3[7, 7] = 1.0
    W3[8, 15] = 1.0
    W3[9, 16] = 1.0

    # Columns containing any marker, including satisfied markers.
    W3[10, 64 + 10] = 1.0
    W3[10, 64 + 11] = 1.0
    W3[11, 64 + 12] = 1.0
    W3[11, 64 + 13] = 1.0

    # Piecewise-linear clock pulses. Each pair (start, end) refers
    # to the last consumed row of the displayed step bar.
    pulses = [(2, 3), (11, 12), (19, 20), (27, 28), (35, 36)]
    clock_coefficients = []
    i = 0
    for start, end in pulses:
        for threshold, coefficient in (
            (start, 63.0),
            (start + 1, -63.0),
            (end, -63.0),
            (end + 1, 63.0),
        ):
            channel = 12 + i
            W3[channel, 64 + 14] = 1.0
            b3[channel] = -float(threshold) / 63.0
            clock_coefficients.append((channel, coefficient))
            i += 1

    # Rays from unsatisfied marker columns.
    for t in range(2):
        right = 2 * t
        left = right + 1
        Wh[right, right, 1, 0] = 1.0
        Wh[left, left, 1, 2] = 1.0
        Uh[right, t] = 1.0
        Uh[left, t] = 1.0

    # Directional hits at goal patches.
    for t in range(2):
        for direction in range(2):
            source = 2 * t + direction
            hit = 4 + source
            Wh[hit, source, 1, 1] = 1.0
            Uh[hit, 2 + t] = 10.0
            Uh[hit, 6 + t] = 0.9
            bh[hit] = -10.9

    # Diffuse directional goal signals.
    for out in (8, 9):
        Wh[out, out, 0, 1] = 0.25
        Wh[out, out, 2, 1] = 0.25
        Wh[out, out, 1, 0] = 0.25
        Wh[out, out, 1, 2] = 0.25

    Wh[8, 4, 1, 1] = 0.4
    Wh[8, 6, 1, 1] = 0.4
    Wh[9, 5, 1, 1] = 0.4
    Wh[9, 7, 1, 1] = 0.4

    # Opposing rays count distinct marker columns.
    for f in (10, 11):
        Uh[12, f] = 1.0
        Uh[13, f] = 1.0
    Wh[12, 12, 1, 0] = 1.0
    Wh[13, 13, 1, 2] = 1.0

    # Single-marker directional button scores.
    for out, signal, button in ((10, 8, 4), (11, 9, 5)):
        Wh[out, signal, 1, 1] = 1.0
        Wh[out, 12, 1, 1] = -100.0
        Wh[out, 13, 1, 1] = -100.0
        Uh[out, 10] = 100.0
        Uh[out, 11] = 100.0
        Uh[out, button] = 100.0

    # Broadcast clock pulses from the left edge.
    Wh[14, 14, 1, 0] = 1.0
    for channel, coefficient in clock_coefficients:
        Uh[14, channel] = coefficient

    # Suppress red buttons during clock pulses when two marker
    # columns are present.
    Wh[15, 14, 1, 1] = 1.0
    Wh[15, 12, 1, 1] = 1.0
    Wh[15, 13, 1, 1] = 1.0
    Uh[15, 10] = -1.0
    Uh[15, 11] = -1.0
    Uh[15, 5] = 1.0
    bh[15] = -3.0

    # ACTION6 is the only requested game action.
    p["bo"][6] = 1.0

    # Directional scores dominate on single-marker layouts. Otherwise,
    # position ranks red controls in one direction and blue controls
    # in the other.
    p["Wc"][0, 4] = 1e-6
    p["Wc"][0, 5] = 1e-6
    p["Wc"][0, 8] = -1e-7
    p["Wc"][0, 9] = 1e-7
    p["Wc"][0, 32 + 10] = 1.0
    p["Wc"][0, 32 + 11] = 1.0
    p["Wc"][0, 32 + 15] = -2e-6

    return p
