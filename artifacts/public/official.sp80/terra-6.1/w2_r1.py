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

    # Coordinate extrema for selected object, cups, source, background.
    for group, colour in enumerate((9, 11, 6, 12)):
        for axis, coord in enumerate((16, 17)):
            plus = 4 * group + 2 * axis
            inv = plus + 1
            p["W1"][plus, colour, 2, 2] = 1
            p["W1"][plus, coord, 2, 2] = 1
            p["b1"][plus] = -1
            p["W1"][inv, colour, 2, 2] = 1
            p["W1"][inv, coord, 2, 2] = -1

    for channel, colour in ((16, 9), (17, 8), (18, 6)):
        p["W1"][channel, colour, 2, 2] = 1

    # Coordinates masked by the unselected-bar colour.
    for axis, coord in enumerate((16, 17)):
        plus = 19 + 2 * axis
        inv = plus + 1
        p["W1"][plus, 8, 2, 2] = 1
        p["W1"][plus, coord, 2, 2] = 1
        p["b1"][plus] = -1
        p["W1"][inv, 8, 2, 2] = 1
        p["W1"][inv, coord, 2, 2] = -1

    # The colour 4 pixel lies immediately upstream of colour 6.
    # Channels encode flow up, down, left, right.
    for ch, dy, dx in (
        (23, 1, 0),
        (24, -1, 0),
        (25, 0, 1),
        (26, 0, -1),
    ):
        p["W1"][ch, 6, 2, 2] = 1
        p["W1"][ch, 4, 2 + dy, 2 + dx] = 1
        p["b1"][ch] = -1

    for i in range(23):
        p["W2"][i, i, 1, 1] = 1

    # Selected-object top, bottom, left and right boundary pixels.
    for out, dy, dx in (
        (23, -1, 0),
        (24, 1, 0),
        (27, 0, -1),
        (28, 0, 1),
    ):
        p["W2"][out, 16, 1, 1] = 1
        p["W2"][out, 16, 1 + dy, 1 + dx] = -1

    # Pass the four source-direction detectors.
    for out, inp in ((25, 23), (26, 24), (29, 25), (30, 26)):
        p["W2"][out, inp, 1, 1] = 1

    for i in range(16):
        p["W3"][i, i] = 1

    # Row and column broadcasts of boundaries and source directions.
    for i in range(4):
        p["W3"][16 + i, 32 + 23 + i] = 1
        p["W3"][24 + i, 64 + 27 + i] = 1

    p["W3"][20, 17] = 1
    p["W3"][21, 19] = 1
    p["W3"][22, 20] = 1
    p["W3"][28, 21] = 1
    p["W3"][29, 22] = 1

    # Propagate selected-object and source rays vertically.
    for h, f, ky in (
        (0, 16, 2), (1, 17, 0),
        (2, 18, 2), (3, 19, 0),
    ):
        p["Wh"][h, h, ky, 1] = 1
        p["Uh"][h, f] = 1

    # Unselected bars intersecting both rays are downstream candidates.
    for h, selected_ray, source_ray in ((4, 0, 2), (5, 1, 3)):
        p["Wh"][h, selected_ray, 1, 1] = 1
        p["Wh"][h, source_ray, 1, 1] = 1
        p["Uh"][h, 20] = 1
        p["bh"][h] = -2.5

    # Prefer the nearest downstream candidate.
    for h, selected_ray, source_ray, feature in (
        (6, 0, 2, 21), (7, 1, 3, 22),
    ):
        p["Wh"][h, selected_ray, 1, 1] = 1
        p["Wh"][h, source_ray, 1, 1] = 1
        p["Uh"][h, feature] = 1
        p["bh"][h] = -2

    # Horizontal counterparts.
    for h, f, kx in (
        (8, 24, 2), (9, 25, 0),
        (10, 26, 2), (11, 27, 0),
    ):
        p["Wh"][h, h, 1, kx] = 1
        p["Uh"][h, f] = 1

    for h, selected_ray, source_ray in ((12, 8, 10), (13, 9, 11)):
        p["Wh"][h, selected_ray, 1, 1] = 1
        p["Wh"][h, source_ray, 1, 1] = 1
        p["Uh"][h, 20] = 1
        p["bh"][h] = -2.5

    for h, selected_ray, source_ray, feature in (
        (14, 8, 10, 28), (15, 9, 11, 29),
    ):
        p["Wh"][h, selected_ray, 1, 1] = 1
        p["Wh"][h, source_ray, 1, 1] = 1
        p["Uh"][h, feature] = 1
        p["bh"][h] = -2

    for h in (4, 5, 12, 13):
        p["Wc"][0, 32 + h] = 10
    for h in (6, 7, 14, 15):
        p["Wc"][0, 32 + h] = 1

    # Affine expressions over g, with the final component as a bias.
    def constant(value):
        v = np.zeros(97, dtype=np.float32)
        v[96] = value
        return v

    def feature(index):
        v = np.zeros(97, dtype=np.float32)
        v[index] = 1
        return v

    def fmax(channel):
        return feature(32 + channel)

    def maximum(group, axis):
        return fmax(4 * group + 2 * axis)

    def minimum(group, axis):
        return constant(1) - fmax(4 * group + 2 * axis + 1)

    def center(group, axis):
        return 0.5 * (maximum(group, axis) + minimum(group, axis))

    def put_z(index, expression):
        p["Wz"][index] = expression[:96]
        p["bz"][index] = expression[96]

    has_bar = fmax(20)
    put_z(0, has_bar - constant(0.5))
    put_z(1, constant(0.5) - has_bar)

    span_row = maximum(0, 0) - minimum(0, 0)
    span_col = maximum(0, 1) - minimum(0, 1)
    length = span_row + span_col - constant(3.0 / 63.0)
    put_z(2, 100 * (constant(0.333) - length))
    put_z(3, 100 * (length - constant(0.333)))

    # Estimate which display axis is parallel to gravity.
    sx = center(2, 1) - center(1, 1)
    sy = center(2, 0) - center(1, 0)
    put_z(50, sx)
    put_z(51, -sx)
    put_z(52, sy)
    put_z(53, -sy)
    p["Wmz"][63, 50:54] = (-50, -50, 50, 50)

    candidate = (
        feature(4) + feature(5) + feature(12) + feature(13)
        - constant(0.25)
    )
    put_z(54, candidate)

    p["bo"][:] = -100
    p["bo"][1:5] = -50
    p["bo"][5] = 1.3
    p["bo"][6] = 0
    p["Wo"][6, 54] = 5.6

    # Restrict movement to the axis perpendicular to gravity.
    p["Wo"][1, 128 + 63] = -50
    p["Wo"][2, 128 + 63] = -50
    p["Wo"][3, 128 + 63] = 50
    p["Wo"][4, 128 + 63] = 50

    next_z = 4
    next_mem = 0

    def add_term(action, expression, coefficient, disable_indices):
        nonlocal next_z, next_mem
        zi = next_z
        next_z += 1
        put_z(zi, expression)

        a = next_mem
        b = next_mem + 1
        next_mem += 2
        p["Wmz"][a, zi] = 1

        for off in disable_indices:
            p["Wmz"][a, off] = -15
            p["Wmz"][b, off] = -15

        p["Wo"][action, 128 + a] += 40 * coefficient
        p["Wo"][action, 128 + b] -= 40 * coefficient

    for axis, negative_action, positive_action in (
        (0, 1, 2),
        (1, 3, 4),
    ):
        selected_center = center(0, axis)
        cup_center = center(1, axis)

        # Single bar: center it over the outer cups.
        add_term(negative_action, selected_center - cup_center, 1, (0,))
        add_term(positive_action, cup_center - selected_center, 1, (0,))

        # Long bars: move toward the nearer edge.
        left_distance = minimum(0, axis) - minimum(3, axis)
        right_distance = maximum(3, axis) - maximum(0, axis)
        long_off = (1, 2)

        add_term(negative_action, left_distance, 1, long_off)
        add_term(
            negative_action, 2 * left_distance - right_distance,
            -1, long_off
        )
        add_term(positive_action, right_distance, 1, long_off)
        add_term(
            positive_action, 2 * right_distance - left_distance,
            -1, long_off
        )

        # Short bars: attract toward intermediate positions between cups.
        other_axis = 1 - axis
        thickness = (
            maximum(0, other_axis) - minimum(0, other_axis)
            + constant(1.0 / 63.0)
        )
        inset = 1.5 * thickness
        a = minimum(1, axis) + inset
        b = maximum(1, axis) - inset
        left_target = (3 * a + b) * 0.25
        right_target = (a + 3 * b) * 0.25
        midpoint = (left_target + right_target) * 0.5
        x = selected_center
        short_off = (1, 3)

        add_term(negative_action, x - right_target, 1, short_off)
        add_term(negative_action, x - left_target, 1, short_off)
        add_term(
            negative_action, x - (left_target + midpoint) * 0.5,
            -2, short_off
        )
        add_term(negative_action, x - midpoint, 1, short_off)

        add_term(positive_action, left_target - x, 1, short_off)
        add_term(positive_action, x - midpoint, 1, short_off)
        add_term(
            positive_action, x - (midpoint + right_target) * 0.5,
            -2, short_off
        )
        add_term(positive_action, x - right_target, 1, short_off)

    return p
