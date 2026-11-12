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

    def template(channel, pixels):
        for y in range(4):
            for x in range(4):
                p["W1"][channel, pixels[y][x], y + 1, x + 1] = 1.0
        p["b1"][channel] = -15.5

    # Player orientations.
    for orientation in range(4):
        pixels = [[14 for _ in range(4)] for _ in range(4)]
        for y in range(4):
            for x in range(4):
                stripe = (
                    (orientation == 0 and y == 0)
                    or (orientation == 1 and x == 3)
                    or (orientation == 2 and y == 3)
                    or (orientation == 3 and x == 0)
                )
                if stripe:
                    pixels[y][x] = 0
        template(orientation, pixels)

    # Player-held, in-front, free, and other-agent-held crates.
    for i, border in enumerate((0, 3, 4, 5)):
        pixels = [
            [
                border if y in (0, 3) or x in (0, 3) else 9
                for x in range(4)
            ]
            for y in range(4)
        ]
        template(4 + i, pixels)

    # Neighborhood presence indicators for colors 2 and 9.
    for channel, color, bias in (
        (8, 2, 0.0),
        (9, 2, -1.0),
        (10, 9, 0.0),
        (11, 9, -1.0),
    ):
        p["W1"][channel, color, :, :] = 1.0
        p["b1"][channel] = bias

    p["W1"][12, 2, 2, 2] = 1.0
    p["W1"][13, 5, 2, 2] = 1.0
    p["W1"][14, 16, 2, 2] = 1.0
    p["W1"][31, 17, 2, 2] = 1.0

    # Direction order: up, down, left, right.
    offsets = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    shifts = [(0, 0), (1, 0), (0, 0), (0, 1)]
    wall_colors = (5, 11, 12, 15)
    full_colors = (3, 4, 5, 11, 12, 15)

    for d, (dy, dx) in enumerate(offsets):
        for base, colors in ((15, wall_colors), (23, full_colors)):
            channel = base + d
            for color in colors:
                p["W1"][channel, color, 2 + dy, 2 + dx] = 1.0
            # Treat pixels outside the observation as blocked.
            p["W1"][channel, 18, 2 + dy, 2 + dx] = -1.0
            p["b1"][channel] = 1.0

        # Neighboring exposed color-2 pixel.
        p["W1"][27 + d, 2, 2 + dy, 2 + dx] = 1.0

    # Patterned non-walkable pixels contain adjacent 2 and 1.
    for d, (dy, dx) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
        channel = 19 + d
        p["W1"][channel, 2, 2, 2] = 1.0
        p["W1"][channel, 1, 2 + dy, 2 + dx] = 1.0
        p["b1"][channel] = -1.5

    # W2 object markers.
    for channel in range(4):
        p["W2"][0, channel, 1, 1] = 1.0
    p["W2"][1, 4, 1, 1] = 1.0
    for channel in (5, 6, 7):
        p["W2"][2, channel, 1, 1] = 1.0
    p["W2"][3, 5, 1, 1] = 1.0

    # Player grid coordinates, gated by the player marker.
    for channel in range(4):
        p["W2"][4, channel, 1, 1] = 32.0
        p["W2"][5, channel, 1, 1] = 32.0
    p["W2"][4, 31, 1, 1] = 63.0 / 4.0
    p["W2"][5, 14, 1, 1] = 63.0 / 4.0
    p["b2"][4] = -16.25
    p["b2"][5] = -16.25

    # Goal-colored pixels near a 9 pixel.
    p["W2"][9, 12, 1, 1] = 1.0
    p["W2"][9, 10, 1, 1] = 1.0
    p["W2"][9, 11, 1, 1] = -1.0
    p["b2"][9] = -1.5

    def marker_and_sampler(out, marker_channels, sampler, dy, dx):
        for channel in marker_channels:
            p["W2"][out, channel, 1, 1] = 2.0
        p["W2"][out, sampler, 1 + dy, 1 + dx] = 1.0
        p["b2"][out] = -1.5

    for d, (dy, dx) in enumerate(shifts):
        marker_and_sampler(10 + d, range(4), 15 + d, dy, dx)
        marker_and_sampler(14 + d, (4,), 23 + d, dy, dx)
        marker_and_sampler(22 + d, range(4), 23 + d, dy, dx)
        marker_and_sampler(26 + d, (4, 5, 6, 7), 27 + d, dy, dx)

    p["W2"][18, 13, 1, 1] = 1.0
    for channel in range(19, 23):
        p["W2"][19, channel, :, :] = 1.0
    p["b2"][19] = -0.25

    # W3 object markers and attraction sources.
    p["W3"][0, 0] = 2.0
    p["W3"][1, 1] = 2.0

    # Exclude crates next to exposed goal-colored pixels.
    p["W3"][2, 2] = 2.0
    p["W3"][3, 2] = 2.0
    for channel in range(26, 30):
        p["W3"][2, channel] = -2.0
        p["W3"][3, channel] = -2.0

    p["W3"][4, 9] = 2.0
    p["W3"][5, 18] = 1.0
    p["W3"][5, 19] = 1.0
    p["W3"][27, 18] = 1.0

    for d in range(4):
        p["W3"][6 + d, 10 + d] = 2.0
        p["W3"][10 + d, 14 + d] = 2.0
        p["W3"][20 + d, 22 + d] = 2.0

    # Grabbable crate in front.
    p["W3"][18, 3] = 2.0
    p["W3"][19, 3] = 2.0
    for channel in range(26, 30):
        p["W3"][18, channel] = -2.0
        p["W3"][19, channel] = -2.0

    # Intersection: held crate AND locally exposed goal color.
    p["W3"][24, 1] = 2.0
    for channel in range(26, 30):
        p["W3"][24, channel] = 2.0
    p["b3"][24] = -1.5

    p["W3"][25, 4] = 1.0
    p["W3"][26, 5] = 1.0

    # Discounted spatial diffusion fields for crates and goals.
    alpha = 0.2499
    for field in (0, 1):
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            p["Wh"][field, field, 1 + dy, 1 + dx] = alpha

    p["Uh"][0, 5] = -10000.0
    # Crates can cross patterned cells, although the player cannot.
    p["Uh"][1, 27] = -10000.0
    p["Uh"][0, 2] = 1.0
    p["Uh"][0, 3] = 1.0
    p["Uh"][1, 4] = 1.0

    # Gate directional field samples at the player or held-crate marker.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    gate = 64.0
    for d, (dy, dx) in enumerate(directions):
        crate_sample = 2 + d
        goal_sample = 6 + d

        p["Wh"][crate_sample, 0, 1 + dy, 1 + dx] = 1.0
        p["Uh"][crate_sample, 0] = gate
        p["bh"][crate_sample] = -gate

        p["Wh"][goal_sample, 1, 1 + dy, 1 + dx] = 1.0
        p["Uh"][goal_sample, 1] = gate
        p["bh"][goal_sample] = -gate

    p["Uh"][10, 24] = 2.0

    # Multiscale clipped directional differences.
    scales = (1.0, 1e2, 1e4, 1e6, 1e8, 1e10)
    opposite = (1, 0, 3, 2)
    for mode in range(2):
        for d in range(4):
            for k, scale in enumerate(scales):
                index = (mode * 4 + d) * len(scales) + k
                plus = 2 * index
                minus = plus + 1
                h_start = 2 if mode == 0 else 6

                for unit in (plus, minus):
                    p["Wz"][unit, h_start + d] = scale
                    p["Wz"][unit, h_start + opposite[d]] = -scale
                p["bz"][minus] = -1.0

                p["Wmz"][index, plus] = 1.0
                p["Wmz"][index, minus] = -1.0
                if mode == 0:
                    p["Wmz"][index, 96] = -10.0
                else:
                    p["Wmz"][index, 96] = 10.0
                    p["bm"][index] = -10.0

    p["Wz"][96, 32 + 1] = 1.0
    p["Wz"][97, 10] = 1.0

    # Grab indicators, gated off while carrying.
    p["Wz"][98, 32 + 18] = 1.0
    p["Wz"][98, 32 + 1] = -1.0
    p["Wz"][99, 32 + 19] = 1.0
    p["Wz"][99, 32 + 1] = -1.0

    p["Wz"][100, 64 + 4] = 4096.0

    for d in range(4):
        p["Wz"][101 + d, 32 + 6 + d] = 1.0

        p["Wz"][105 + d, 32 + 10 + d] = 2.0
        p["Wz"][105 + d, 32 + 1] = 2.0
        p["bz"][105 + d] = -3.0

        p["Wz"][109 + d, 32 + 20 + d] = 2.0
        p["Wz"][109 + d, 32 + 1] = 2.0
        p["bz"][109 + d] = -3.0

    p["Wz"][113, 32 + 25] = 1.0
    p["Wz"][114, 32 + 26] = 1.0

    # Detect disappearance of the last visible goal-colored source.
    p["Wmz"][48, 100] = 0.01
    p["Wm"][49, 48] = 20.0
    p["Wmz"][49, 100] = -20.0
    p["Wmz"][49, 96] = 20.0
    p["Wma"][49, 5] = -20.0
    p["bm"][49] = -20.05

    # Store the player coordinates at this step.
    p["Wmz"][50, 113] = 0.001
    p["Wmz"][51, 114] = 0.001

    # For each direction, detect whether that previous move failed.
    for d, action in enumerate((1, 2, 3, 4)):
        unit = 52 + d
        is_vertical = d < 2
        current_coord = 114 if is_vertical else 113
        previous_coord = 51 if is_vertical else 50
        sign = 1.0 if d in (0, 2) else -1.0

        p["Wmz"][unit, current_coord] = 5.0 * sign
        p["Wm"][unit, previous_coord] = -5000.0 * sign
        p["Wma"][unit, action] = 50.0
        p["bm"][unit] = -47.5

        # Previous action indicator for a mild reversal penalty.
        p["Wma"][56 + d, action] = 10.0
        p["bm"][56 + d] = -5.0

    for d, action in enumerate((1, 2, 3, 4)):
        p["bo"][action] = 10.0
        for mode in range(2):
            for k in range(len(scales)):
                index = (mode * 4 + d) * len(scales) + k
                p["Wo"][action, 128 + index] = 1.0

        p["Wo"][action, 101 + d] = -100.0
        p["Wo"][action, 105 + d] = -100.0
        p["Wo"][action, 109 + d] = -100.0
        p["Wo"][action, 128 + 52 + d] = -10.0
        p["Wo"][action, 128 + 56 + opposite[d]] = -1.0

    p["Wo"][5, 97] = 100.0
    p["Wo"][5, 98] = 50.0
    p["Wo"][5, 99] = 50.0
    p["Wo"][5, 128 + 49] = 20.0
    p["bo"][5] = 5.0

    p["bo"][0] = -1000.0
    p["bo"][6] = -1000.0
    p["bo"][7] = -1000.0
    return p
