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
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def c1(out, inp, dy=0, dx=0, value=1.0):
        p["W1"][out, inp, dy + 2, dx + 2] += value

    def c2(out, inp, dy=0, dx=0, value=1.0):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    def covered_marker(out, colors, dy=0, dx=0):
        neighbours = []
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                if oy == 0 and ox == 0:
                    continue
                yy, xx = dy + oy, dx + ox
                if -2 <= yy <= 2 and -2 <= xx <= 2:
                    neighbours.append((yy, xx))

        n = len(neighbours)
        c1(out, 11, dy, dx, float(n))
        for yy, xx in neighbours:
            for color in colors:
                c1(out, color, yy, xx)
        p["b1"][out] = -(2 * n - 1)

    # Selection, goals, coordinates, energy-column exclusion, reflections.
    c1(0, 0)
    c1(3, 11)
    c1(4, 16)
    c1(5, 17)
    c1(6, 17, value=126.0)
    p["b1"][6] = -125.0
    c1(7, 4)

    # Selected vertical/horizontal line markers.
    c1(1, 0)
    c1(1, 10, -2, 0)
    c1(1, 10, 2, 0)
    p["b1"][1] = -2.0

    c1(2, 0)
    c1(2, 10, 0, -2)
    c1(2, 10, 0, 2)
    p["b1"][2] = -2.0

    # First two pixels of three-pixel translation probes.
    for d, (dy, dx) in enumerate(directions):
        yy, xx = -2 * dy, -2 * dx
        c1(8 + d, 0, yy, xx)
        c1(12 + d, 4, yy, xx)
        covered_marker(16 + d, (0, 5), yy, xx)
        covered_marker(27 + d, (4,), yy, xx)

    covered_marker(24, (0, 5))
    covered_marker(25, (0, 4, 5))
    covered_marker(26, (4,))

    # Dense goal seed: a filled two-by-two cell patch.
    for yy, xx in ((-1, -1), (-1, 2), (2, -1), (2, 2)):
        c1(20, 11, yy, xx)
    p["b1"][20] = -3.0

    # Dense selected-object seed. Any surviving selection marker suffices.
    for yy, xx in ((-2, -2), (-2, 1), (1, -2), (1, 1)):
        c1(21, 5, yy, xx, 4.0)
    for yy, xx in ((-1, -1), (-1, 2), (2, -1), (2, 2)):
        c1(21, 0, yy, xx)
    p["b1"][21] = -16.0

    c2(0, 0)
    c2(1, 1)
    c2(2, 2)

    # Goal extrema: row, inverse row, column, inverse column.
    for out, coord, sign in (
        (3, 4, 1.0),
        (4, 4, -1.0),
        (5, 5, 1.0),
        (6, 5, -1.0),
    ):
        c2(out, 3)
        c2(out, coord, value=sign)
        c2(out, 6, value=-2.0)
        p["b2"][out] = -1.0 if sign > 0 else 0.0

    c2(7, 1)
    c2(7, 5)
    p["b2"][7] = -1.0
    c2(8, 2)
    c2(8, 4)
    p["b2"][8] = -1.0

    # Complete each translation with one additional pixel.
    for d, (dy, dx) in enumerate(directions):
        c2(9 + d, 8 + d, -dy, -dx)
        c2(9 + d, 16 + d, -dy, -dx)
        c2(13 + d, 12 + d, -dy, -dx)
        c2(13 + d, 27 + d, -dy, -dx)
        c2(21 + d, 8 + d, -dy, -dx)

    c2(17, 7)
    c2(17, 26)

    # Restore covered goals to their original three-by-three footprints.
    c2(18, 3)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                c2(18, 25, dy, dx)
    c2(18, 6, value=-2.0)

    c2(19, 20)
    c2(20, 21)
    c2(25, 0)
    c2(25, 24)

    def f3(out, inp, value=1.0, section=0):
        # Sections: local=0, row maximum=32, column maximum=64.
        p["W3"][out, section + inp] += value

    f3(0, 18)

    # Four candidate translations and the current placement.
    for d in range(5):
        marker = 21 + d if d < 4 else 0
        original = 9 + d if d < 4 else 25

        # Difference of these channels is a clipped axial ownership union.
        for out, bias in ((1 + d, -2.0), (6 + d, -3.0)):
            f3(out, original, 2.0)
            f3(out, marker, section=32)
            f3(out, marker, section=64)
            p["b3"][out] = bias

        if d < 4:
            vd = (0, 1, 3, 2)[d]
            hd = (1, 0, 2, 3)[d]
            vertical_body = 13 + vd
            horizontal_body = 13 + hd
        else:
            vertical_body = horizontal_body = 17

        f3(11 + d, vertical_body)
        f3(11 + d, marker, section=32)
        p["b3"][11 + d] = -1.0

        f3(16 + d, horizontal_body)
        f3(16 + d, marker, section=64)
        p["b3"][16 + d] = -1.0

    f3(21, 1)
    f3(22, 2)
    for out, inp in (
        (23, 3), (24, 4), (25, 5), (26, 6),
        (27, 7), (28, 8),
    ):
        f3(out, inp)

    f3(29, 20)
    f3(30, 19)

    # h1-h2: capped broadcast of the selected dense-shape indicator.
    cross = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    for out in (1, 2):
        for dy, dx in cross:
            p["Wh"][out, 1, dy + 1, dx + 1] = 1.0
            p["Wh"][out, 2, dy + 1, dx + 1] = -1.0
        p["Uh"][out, 29] = 1.0
    p["bh"][2] = -1.0

    # h3-h4: capped flood fill of dense seeds inside connected goal regions.
    for out in (3, 4):
        for dy, dx in cross:
            p["Wh"][out, 3, dy + 1, dx + 1] = 1.0
            p["Wh"][out, 4, dy + 1, dx + 1] = -1.0
        p["Uh"][out, 0] = 10.0
        p["Uh"][out, 30] = 1.0
    p["bh"][3] = -10.0
    p["bh"][4] = -11.0

    # h5: dense goals activated by a dense selected shape.
    p["Wh"][5, 1, 1, 1] = 1.0
    p["Wh"][5, 2, 1, 1] = -1.0
    p["Wh"][5, 3, 1, 1] = 1.0
    p["Wh"][5, 4, 1, 1] = -1.0
    p["bh"][5] = -1.0

    # h0: discounted goal diffusion, with structural target preference.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                p["Wh"][0, 0, dy + 1, dx + 1] = 0.12
    p["Uh"][0, 0] = 1.0 / 64.0
    p["Wh"][0, 3, 1, 1] = -1.0 / 64.0
    p["Wh"][0, 4, 1, 1] = 1.0 / 64.0
    p["Wh"][0, 5, 1, 1] = 1.0

    # Paired probes clip the union of original and reflected masks.
    # Their difference samples h0 without double-counting intersections.
    gate = 32.0
    for d in range(5):
        a = 6 + 2 * d
        b = a + 1
        for out in (a, b):
            p["Uh"][out, 1 + d] = gate
            p["Uh"][out, 6 + d] = -gate
            p["Uh"][out, 11 + d] = gate
            p["Uh"][out, 16 + d] = gate
            p["bh"][out] = -gate
        p["Wh"][a, 0, 1, 1] = 1.0

    def hmean(channel):
        return 16 + channel

    def fmax(channel):
        return 32 + channel

    def fmean(channel):
        return 64 + channel

    # Positive score improvements over the current placement.
    for d in range(4):
        a = 6 + 2 * d
        b = a + 1
        p["Wz"][d, hmean(a)] = 1.0
        p["Wz"][d, hmean(b)] = -1.0
        p["Wz"][d, hmean(14)] = -1.0
        p["Wz"][d, hmean(15)] = 1.0

    # Count-based gates distinguish selected lines from isolated crossings.
    count_scale = 4096.0
    for unit, vertical in ((8, True), (9, False)):
        own, other = (21, 22) if vertical else (22, 21)
        p["Wz"][unit, fmean(own)] = 0.5 * count_scale
        p["Wz"][unit, fmean(other)] = -0.5 * count_scale
        p["bz"][unit] = -4.0

    # A paired-ReLU gate exposes signed alignment errors only for lines.
    dead_band = 1.1 / 63.0
    for unit, axis, maximum, inverse, vertical, sign in (
        (4, 28, 23, 24, False, 1.0),
        (5, 28, 23, 24, False, -1.0),
        (6, 27, 25, 26, True, 1.0),
        (7, 27, 25, 26, True, -1.0),
    ):
        own, other = (21, 22) if vertical else (22, 21)
        p["Wz"][unit, fmean(own)] = 0.5 * count_scale
        p["Wz"][unit, fmean(other)] = -0.5 * count_scale
        p["Wz"][unit, fmax(axis)] = sign
        p["Wz"][unit, fmax(maximum)] = -0.5 * sign
        p["Wz"][unit, fmax(inverse)] = 0.5 * sign
        p["bz"][unit] = -4.0 - 0.5 * sign - dead_band

    # Clipped line-presence indicators suppress the shape policy on lines.
    for first, vertical in ((10, True), (12, False)):
        own, other = (21, 22) if vertical else (22, 21)
        for unit, threshold in ((first, 8.0), (first + 1, 9.0)):
            p["Wz"][unit, fmean(own)] = count_scale
            p["Wz"][unit, fmean(other)] = -count_scale
            p["bz"][unit] = -threshold

    p["bo"][:] = -1.0e9
    p["bo"][1:5] = 0.0
    p["bo"][5] = 0.25

    for d in range(4):
        action = 1 + d
        p["Wo"][action, d] = 1.0e8
        p["Wo"][action, 4 + d] = 1.0e11
        baseline = 9 if d < 2 else 8
        p["Wo"][action, baseline] = -1.0e11

        p["Wo"][action, 10] = -1.0e9
        p["Wo"][action, 11] = 1.0e9
        p["Wo"][action, 12] = -1.0e9
        p["Wo"][action, 13] = 1.0e9

    # Immediate previous-action memory discourages short oscillations.
    for action in range(8):
        p["Wma"][action, action] = 10.0
    for action, previous in ((1, 2), (2, 1), (3, 4), (4, 3)):
        p["Wo"][action, 128 + previous] = -1.0e7

    return p
