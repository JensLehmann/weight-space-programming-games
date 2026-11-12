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

    colors = (7, 10, 11, 14)
    button_colors = (7, 10, 11, 14, 8, 12)

    def detector(out, conditions):
        for dy, dx, accepted in conditions:
            for color in accepted:
                p["W1"][out, color, 2 + dy, 2 + dx] += 2.0
        p["b1"][out] = 1.0 - 2.0 * len(conditions)

    # Marker centres, classified by surrounding arm colour.
    for i, color in enumerate(colors):
        conditions = [(0, 0, (13,))]
        conditions += [
            (dy, dx, (color, 3))
            for dy, dx in ((-1, -1), (-1, 1), (1, -1), (1, 1))
        ]
        detector(i, conditions)

    # Pivot caps: left, right, top, bottom.
    arm_colors = (0, 6, 7, 8, 9, 10, 11, 12, 13, 14)
    detector(4, [(0, 0, (3,)), (-1, 0, (3,)), (1, 0, (3,)),
                 (0, 1, arm_colors)])
    detector(5, [(0, 0, (3,)), (-1, 0, (3,)), (1, 0, (3,)),
                 (0, -1, arm_colors)])
    detector(6, [(0, 0, (3,)), (0, -1, (3,)), (0, 1, (3,)),
                 (1, 0, arm_colors)])
    detector(7, [(0, 0, (3,)), (0, -1, (3,)), (0, 1, (3,)),
                 (-1, 0, arm_colors)])

    # Target cross and coordinate planes.
    detector(8, [(-1, 0, (13,)), (1, 0, (13,)),
                 (0, -1, (13,)), (0, 1, (13,))])
    p["W1"][9, 17, 2, 2] = 1.0
    p["W1"][10, 16, 2, 2] = 1.0

    # Compact button detectors for small and large horizontal controls.
    for i, color in enumerate(button_colors):
        c = (color,)
        common = [(0, 0, c), (0, 1, c), (0, 2, c)]
        detector(11 + 2 * i,
                 common + [(0, -1, (2, 4)), (-1, 0, (2, 4))])
        detector(12 + 2 * i,
                 common + [(0, -2, (3,)), (-1, 0, (2, 4))])

    # Bottom half of the vertical colour-11 control.
    detector(23, [(0, 0, (11,)), (0, 1, (11,)), (0, 2, (11,)),
                  (0, -1, (2, 4)), (-1, 0, (11,))])
    p["W1"][28, 2, 2, 4] = 1.0

    # Local obstruction detectors: marker on colour 7 near colour 12,
    # or marker on colour 10 near colour 8.
    block_specs = [
        (24, 7, 12, 0, 2),
        (25, 7, 12, 2, 0),
        (26, 7, 12, 0, -2),
        (27, 7, 12, -2, 0),
        (29, 10, 8, 0, 2),
        (30, 10, 8, 2, 0),
        (31, 10, 8, 0, -2),
    ]
    for out, color, blocker, dy, dx in block_specs:
        conditions = [(0, 0, (13,)), (dy, dx, (blocker,))]
        conditions += [
            (yy, xx, (color, 3))
            for yy, xx in ((-1, -1), (-1, 1), (1, -1), (1, 1))
        ]
        detector(out, conditions)

    # Uncovered markers; blocked markers are handled separately.
    for i in range(4):
        p["W2"][i, i, 1, 1] = 1.0
        p["W2"][i, 8, 1, 1] = -1.0
    for ch in (24, 25, 26, 27):
        p["W2"][0, ch, 1, 1] = -1.0
    for ch in (29, 30, 31):
        p["W2"][1, ch, 1, 1] = -1.0

    for i in range(4, 8):
        p["W2"][i, i, 1, 1] = 1.0

    # Uncovered target centres.
    p["W2"][8, 8, 1, 1] = 1.0
    for i in range(4):
        p["W2"][8, i, 1, 1] = -1.0

    p["W2"][9, 9, 1, 1] = 1.0
    p["W2"][10, 10, 1, 1] = 1.0

    # Target x, target y, 1-target x, and 1-target y.
    for out, coord, sign, bias in (
        (11, 9, 1.0, -2.0),
        (12, 10, 1.0, -2.0),
        (25, 9, -1.0, -1.0),
        (26, 10, -1.0, -1.0),
    ):
        p["W2"][out, coord, 1, 1] = sign
        p["W2"][out, 8, 1, 1] = 2.0
        for i in range(4):
            p["W2"][out, i, 1, 1] = -2.0
        p["b2"][out] = bias

    # Main-colour control halves.
    for i in range(4):
        p["W2"][13 + 2 * i, 11 + 2 * i, 1, 1] = 1.0
        p["W2"][14 + 2 * i, 12 + 2 * i, 1, 1] = 1.0

    # Validate the lower half of the vertical colour-11 control.
    p["W2"][23, 23, 1, 0] = 1.0
    p["W2"][23, 28, 1, 2] = 1.0
    p["b2"][23] = -1.0

    # Blocked-marker indicators.
    for ch in (24, 25, 26, 27):
        p["W2"][27, ch, 1, 1] = 1.0
    p["W2"][27, 8, 1, 1] = -1.0

    for ch in (29, 30, 31):
        p["W2"][28, ch, 1, 1] = 1.0
    p["W2"][28, 8, 1, 1] = -1.0

    # Shorten halves for the two blocker colours.
    p["W2"][29, 19, 1, 1] = 1.0  # colour 8
    p["W2"][30, 21, 1, 1] = 1.0  # colour 12

    # Four signed displacement features, two validity baselines,
    # and two button features per marked-arm colour.
    for i in range(4):
        base = 8 * i
        for direction in range(4):
            out = base + direction
            horizontal = direction < 2
            axial = 32 if horizontal else 64

            p["W3"][out, i] = 10.0
            p["W3"][out, axial + 4 + direction] = 10.0
            p["W3"][out, axial + 8] = 10.0

            local_coord = 9 if horizontal else 10
            if direction in (0, 2):
                inverse_goal = 25 if horizontal else 26
                p["W3"][out, axial + inverse_goal] = -1.0
                p["W3"][out, local_coord] = -1.0
                p["b3"][out] = -28.0
            else:
                goal = 11 if horizontal else 12
                p["W3"][out, local_coord] = 1.0
                p["W3"][out, axial + goal] = -1.0
                p["b3"][out] = -29.0

        for axis in range(2):
            out = base + 4 + axis
            axial = 32 if axis == 0 else 64
            caps = (4, 5) if axis == 0 else (6, 7)
            p["W3"][out, i] = 3.0
            p["W3"][out, axial + 8] = 1.0
            for cap in caps:
                p["W3"][out, axial + cap] = 1.0
            p["b3"][out] = -4.0

        if i == 0:
            p["W3"][base + 4, 27] = 3.0
            p["W3"][base + 5, 27] = 3.0
            p["W3"][base + 6, 30] = 1.0
        elif i == 1:
            p["W3"][base + 4, 28] = 3.0
            p["W3"][base + 5, 28] = 3.0
            p["W3"][base + 6, 29] = 1.0
        else:
            p["W3"][base + 6, 13 + 2 * i] = 1.0

        p["W3"][base + 7, 14 + 2 * i] = 1.0
        if i == 2:
            p["W3"][base + 7, 23] = 1.0

    # Diffuse adjustment signals, then gate them at matching controls.
    for i in range(4):
        f = 8 * i
        extend, shrink, click_extend, click_shrink = range(4 * i, 4 * i + 4)

        for j in range(4):
            p["Uh"][extend, f + j] = 0.0001
            p["Uh"][shrink, f + j] = -0.0001
        for j in (4, 5):
            p["Uh"][extend, f + j] = -0.000102
            p["Uh"][shrink, f + j] = 0.000098

        for wave in (extend, shrink):
            for dy, dx in ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2)):
                p["Wh"][wave, wave, dy, dx] = 0.2019

        gate = 0.002
        p["Wh"][click_extend, extend, 1, 1] = 1.0
        p["Uh"][click_extend, f + 7] = gate
        p["bh"][click_extend] = -gate

        p["Wh"][click_shrink, shrink, 1, 1] = 1.0
        p["Uh"][click_shrink, f + 6] = gate
        p["bh"][click_shrink] = -gate

        p["Wc"][0, 32 + click_extend] = 1.0
        p["Wc"][0, 32 + click_shrink] = 1.0

    p["bo"][6] = 1.0
    return p
