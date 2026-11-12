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

    groups = (
        (8, 10, 13, 15),
        (9, 11, 12, 14),
    )
    palette = groups[0] + groups[1]

    directions = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),            (1, 0),
        (-1, 1),  (0, 1),  (1, 1),
    ]

    def add_colours(channel, colours, dx, dy, value=1.0):
        for colour in colours:
            p["W1"][channel, colour, dy + 2, dx + 2] += value

    # For each clue direction, detect the two cases requiring group 0,
    # plus the presence of any valid direction mark.
    for d, (dx, dy) in enumerate(directions):
        a, b, valid = 3 * d, 3 * d + 1, 3 * d + 2

        add_colours(a, groups[0], 0, 0)
        add_colours(a, (0,), 2 * dx, 2 * dy)
        p["b1"][a] = -1

        add_colours(b, groups[1], 0, 0)
        add_colours(b, (2, 3), 2 * dx, 2 * dy)
        p["b1"][b] = -1

        add_colours(valid, palette, 0, 0)
        add_colours(valid, (0, 2, 3), 2 * dx, 2 * dy)
        p["b1"][valid] = -1

    # Detect ordinary and cross-switch tiles in each colour group.
    for special in range(2):
        for bit in range(2):
            channel = 24 + 2 * special + bit
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    arm = abs(dx) + abs(dy) == 1
                    colours = (6,) if special and arm else groups[bit]
                    add_colours(channel, colours, 2 * dx, 2 * dy)
            p["b1"][channel] = -8

    # Separator tests. W2 shifts these tests by one pixel, so together
    # they select the bottom-right pixel of a doubled centre pixel.
    add_colours(28, (4,), 2, 0)
    add_colours(29, (4,), 0, 2)

    def require_separators(out):
        p["W2"][out, 28, 1, 2] = 1
        p["W2"][out, 29, 2, 1] = 1
        p["b2"][out] = -2

    # Produce a single group-0 or group-1 requirement per clue direction.
    for d in range(8):
        a, b, valid = 3 * d, 3 * d + 1, 3 * d + 2
        out0, out1 = 2 * d, 2 * d + 1

        p["W2"][out0, a, 1, 1] = 1
        p["W2"][out0, b, 1, 1] = 1
        require_separators(out0)

        p["W2"][out1, valid, 1, 1] = 1
        p["W2"][out1, a, 1, 1] = -1
        p["W2"][out1, b, 1, 1] = -1
        require_separators(out1)

    for i in range(4):
        out = 16 + i
        p["W2"][out, 24 + i, 1, 1] = 1
        require_separators(out)

    for i in range(20):
        p["W3"][i, i] = 1

    # Propagate requirements eight display pixels in their respective
    # directions. A tile already in the required group suppresses them.
    for d, (dx, dy) in enumerate(directions):
        for bit in range(2):
            channel = 2 * d + bit
            p["Wh"][channel, channel, 1 - dy, 1 - dx] = 1
            p["Uh"][channel, channel] = 1
            p["Uh"][channel, 16 + bit] = -2
            p["Uh"][channel, 18 + bit] = -2
            p["bh"][channel] = -0.1

    # Restrict clicks to detected tile centres, and rank those centres
    # by their incoming unsatisfied requirements.
    p["Wc"][0, 16:18] = 100
    p["Wc"][0, 18:20] = 100.01
    p["Wc"][0, 32:48] = 1

    p["bo"][6] = 1
    return p
