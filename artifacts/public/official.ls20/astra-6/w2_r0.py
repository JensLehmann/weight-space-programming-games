import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,),
        "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,),
        "Wh": (16, 16, 3, 3), "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,),
        "Wm": (64, 64), "Wmz": (64, 128),
        "Wma": (64, 8), "bm": (64,),
        "Wo": (8, 192), "bo": (8,),
        "Wc": (1, 48), "bc": (1,),
    }
    p = {name: np.zeros(shape, dtype=np.float32)
         for name, shape in shapes.items()}

    colours = (12, 9, 14, 8)
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def c1(o, i, dy=0, dx=0, value=1):
        p["W1"][o, i, dy + 2, dx + 2] += value

    def c2(o, i, dy=0, dx=0, value=1):
        p["W2"][o, i, dy + 1, dx + 1] += value

    def ch(o, i, dy=0, dx=0, value=1):
        p["Wh"][o, i, dy + 1, dx + 1] += value

    # Basic colour features.
    for code, colour in enumerate(colours):
        c1(0, colour)
        c1(1, colour, value=code)
        c1(27, colour, value=code * code)
    c1(2, 4)

    # Distances from the point used to sample the doubled HUD glyph.
    for channel, coordinate, scale, bias in (
        (3, 16, 63, -57),
        (4, 16, -63, 57),
        (5, 17, 63, -5),
        (6, 17, -63, 5),
    ):
        c1(channel, coordinate, value=scale)
        p["b1"][channel] = bias

    # Four shifted occupancy planes suffice for all nine HUD samples.
    shifted = {}
    for channel, (dy, dx) in enumerate(
        ((-1, -1), (-1, 1), (1, -1), (1, 1)), start=7
    ):
        shifted[(dy, dx)] = channel
        for colour in colours:
            c1(channel, colour, dy, dx)

    # Outside-distance features for the complete HUD glyph rectangle.
    for channel, coordinate, scale, bias in (
        (11, 16, 63, -60),
        (12, 16, -63, 55),
        (13, 17, 63, -8),
        (14, 17, -63, 3),
    ):
        c1(channel, coordinate, value=scale)
        p["b1"][channel] = bias

    # Goal-centre detector: a black perimeter and coloured inner pixels.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if abs(dy) == 2 or abs(dx) == 2:
                c1(15, 5, dy, dx, 9)
            else:
                for colour in colours:
                    c1(15, colour, dy, dx)
    p["b1"][15] = -144

    c1(16, 16, value=63)
    p["b1"][16] = -54

    # Exact player-centre detector.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            c1(17, 12 if dy < 0 else 9, dy, dx)
    p["b1"][17] = -24

    # Shape, rotation, and colour tools.
    tool_templates = {
        18: ((0, 3, 3), (3, 0, 0), (3, 0, 3)),
        19: ((3, 0, 3), (1, 0, 0), (3, 1, 3)),
        20: ((9, 14, 14), (9, 0, 8), (12, 12, 8)),
    }
    for channel, template in tool_templates.items():
        for y in range(3):
            for x in range(3):
                c1(channel, template[y][x], y - 1, x - 1)
        p["b1"][channel] = -8

    # Refill detector.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                c1(21, 11, dy, dx)
    p["b1"][21] = -7

    # Samples at the boundary of the player's next cell.
    for a, (dy, dx) in enumerate(directions):
        c1(22 + a, 4, 2 * dy, 2 * dx)
        c1(22 + a, 1, 2 * dy, 2 * dx)

    # Horizontal energy-bar run.
    for dx in range(-2, 3):
        c1(26, 11, 0, dx)
    p["b1"][26] = -4

    # Straight blue pusher boundaries, expanded into their unsafe side.
    for dx in (-2, 0, 2):
        for dy in range(-2, 3):
            c1(29, 1, dy, dx)
    p["b1"][29] = -2
    for dy in (-2, 0, 2):
        for dx in range(-2, 3):
            c1(30, 1, dy, dx)
    p["b1"][30] = -2

    def hud_point_gate(output):
        for coordinate in (3, 4, 5, 6):
            c2(output, coordinate, value=-100)

    bit = 0
    for dy in (-2, 0, 2):
        for dx in (-2, 0, 2):
            if dy == 0 and dx == 0:
                c2(bit, 0)
            else:
                sy = -1 if dy < 0 else 1
                sx = -1 if dx < 0 else 1
                c2(bit, shifted[(sy, sx)], dy - sy, dx - sx)
            hud_point_gate(bit)
            bit += 1

    # Direct HUD colour and squared-colour values.
    # Spatial maxima are exactly c and c^2.
    c2(18, 1)
    c2(19, 27)
    for output in (18, 19):
        for coordinate in (11, 12, 13, 14):
            c2(output, coordinate, value=-100)

    # Shared ranking baseline keeps goal descriptor maxima together.
    # Smaller magnitudes reduce cancellation in the symbolic comparisons.
    def goal_base(output):
        for source, weight in (
            (15, 256), (3, -4), (4, 4),
            (5, -0.25), (6, 0.25), (16, -20000),
        ):
            c2(output, source, value=weight)
        p["b2"][output] = -32 - 4 * 57 - 0.25 * 5

    bit = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            goal_base(9 + bit)
            c2(9 + bit, 0, dy, dx)
            bit += 1

    goal_base(20)
    goal_base(21)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            c2(21, 1, dy, dx, 0.01)

    # Navigation barriers.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            c2(22, 2, dy, dx)
            c2(22, 29, dy, dx)
            c2(22, 30, dy, dx)
    c2(22, 6)
    c2(22, 16)

    # The timer signal is weak enough not to activate player readouts.
    c2(23, 17)
    c2(23, 26, value=0.1)

    c2(24, 21, value=10)
    c2(25, 18, value=10)
    c2(26, 20, value=10)
    c2(27, 19, value=10)

    # Player-local boundary bits combined with normalized coordinates.
    for a, (dy, dx) in enumerate(directions):
        output = 28 + a
        c2(output, 17, value=100)
        c2(output, 22 + a, dy, dx, 2)
        if a in (0, 2):
            c2(output, 5, value=1 / 64)
            c2(output, 6, value=-1 / 64)
            p["b2"][output] = -100 + 5 / 64
        else:
            c2(output, 3, value=1 / 64)
            c2(output, 4, value=-1 / 64)
            p["b2"][output] = -100 + 57 / 64

    p["W3"][:, :32] = np.eye(32, dtype=np.float32)

    # Five recurrent navigation fields:
    # goal, shape tool, colour tool, rotation tool, refill.
    # A pair of channels represents capped reachability by subtraction.
    # The third integrates an encoded horizontal/vertical arrival gradient.
    sources = (20, 25, 26, 27, 24)
    neighbours = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))

    for field, source in enumerate(sources):
        base = 3 * field
        for output in (base, base + 1):
            for dy, dx in neighbours:
                ch(output, base, dy, dx)
                ch(output, base + 1, dy, dx, -1)
            p["Uh"][output, source] = 0.01 if field == 0 else 1
            p["Uh"][output, 22] = -100
        p["bh"][base + 1] = -1

        accumulator = base + 2
        ch(accumulator, accumulator)
        p["Uh"][accumulator, 23] = 100
        p["bh"][accumulator] = -90

        for dy, dx, weight in (
            (0, 1, 3), (0, -1, -3),
            (1, 0, 1), (-1, 0, -1),
        ):
            ch(accumulator, base, dy, dx, weight)
            ch(accumulator, base + 1, dy, dx, -weight)

    # Arrival-time accumulator for refill-distance estimation.
    ch(15, 15)
    ch(15, 12)
    ch(15, 13, value=-1)
    p["Uh"][15, 23] = 2
    p["bh"][15] = -2

    # Decode the nine ordinary values of 3*gx + gy.
    knots = np.arange(-8, 9, 2, dtype=np.float32)
    cases = (
        (-2, -2), (-2, 0), (-2, 2),
        (0, -2), (0, 0), (0, 2),
        (2, -2), (2, 0), (2, 2),
    )
    nav_vectors = []
    nav_constants = []

    for field in range(5):
        for j, knot in enumerate(knots):
            unit = 9 * field + j
            p["Wz"][unit, 3 * field + 2] = 1
            p["bz"][unit] = -1280 - knot

        vectors, constants = [], []
        for a in range(4):
            values = np.array([
                float((gy < 0, gy > 0, gx < 0, gx > 0)[a])
                for gx, gy in cases
            ], dtype=np.float32)
            slopes = np.diff(values) / 2
            coefficients = np.empty(9, dtype=np.float32)
            coefficients[0] = slopes[0]
            coefficients[1:8] = slopes[1:] - slopes[:-1]
            coefficients[8] = -slopes[-1]

            vector = np.zeros(128, dtype=np.float32)
            vector[9 * field:9 * field + 9] = coefficients
            vectors.append(vector)
            constants.append(float(values[0]))
        nav_vectors.append(vectors)
        nav_constants.append(constants)

    cursor = 45

    def binary_difference(raw_difference, tolerance=0.25):
        """Four ReLUs: zero near equality, one for integer inequality."""
        nonlocal cursor
        result = np.zeros(128, dtype=np.float32)
        for sign, threshold, coefficient in (
            (1, tolerance, 2),
            (1, tolerance + 0.5, -2),
            (-1, tolerance, 2),
            (-1, tolerance + 0.5, -2),
        ):
            p["Wz"][cursor] = sign * raw_difference
            p["bz"][cursor] = -threshold
            result[cursor] = coefficient
            cursor += 1
        return result

    # Rotation-invariant shape signatures:
    # 5*(occupied count) + 10*(centre bit) + (corner count).
    # These distinguish all six available shapes.
    signature = np.full(9, 5, dtype=np.float32)
    signature[4] += 10
    signature[[0, 2, 6, 8]] += 1

    raw_shape_difference = np.zeros(96, dtype=np.float32)
    raw_shape_difference[32:41] = signature
    raw_shape_difference[41:50] = -signature
    raw_shape_difference[52] = signature.sum()
    shape_error = binary_difference(raw_shape_difference)

    # A binary code compares the complete oriented glyph directly.
    binary_code = 2.0 ** np.arange(9, dtype=np.float32)
    raw_pixel_difference = np.zeros(96, dtype=np.float32)
    raw_pixel_difference[32:41] = binary_code
    raw_pixel_difference[41:50] = -binary_code
    raw_pixel_difference[52] = binary_code.sum()
    pixel_error = binary_difference(raw_pixel_difference)

    # Current-colour-conditioned comparison with the goal's colour sum.
    # The quadratic gate is linear in the supplied c and c^2 features.
    colour_error = np.zeros(128, dtype=np.float32)
    for code in range(4):
        difference = np.zeros(96, dtype=np.float32)
        difference[53] = 100
        difference[52] = -100 + 9 * code
        difference[41:50] = -code

        gate = np.zeros(96, dtype=np.float32)
        gate[50] = 200 * code
        gate[51] = -100
        gate_bias = -100 * code * code

        for sign, threshold, coefficient in (
            (1, 0.25, 2),
            (1, 0.75, -2),
            (-1, 0.25, 2),
            (-1, 0.75, -2),
        ):
            p["Wz"][cursor] = sign * difference + gate
            p["bz"][cursor] = gate_bias - threshold
            colour_error[cursor] = coefficient
            cursor += 1

    # Decode wall indicators while preserving the coordinate features.
    raw_units, wall_plus, wall_minus = [], [], []
    for a in range(4):
        raw, plus, minus = cursor, cursor + 1, cursor + 2
        raw_units.append(raw)
        wall_plus.append(plus)
        wall_minus.append(minus)
        for unit, offset in ((raw, 0), (plus, -1), (minus, -2)):
            p["Wz"][unit, 60 + a] = 1
            p["bz"][unit] = offset
        cursor += 3

    # The mean of the player/timer plane gives remaining energy.
    energy = np.zeros(96, dtype=np.float32)
    energy[64 + 23] = 5 * 4096
    energy[32 + 23] = -5
    energy_bias = 4

    energy_unit = cursor
    urgent_unit = cursor + 1
    full_unit = cursor + 2
    no_refill_unit = cursor + 3
    cursor += 4

    p["Wz"][energy_unit] = energy
    p["bz"][energy_unit] = energy_bias

    p["Wz"][urgent_unit] = -energy
    p["Wz"][urgent_unit, 15] = -0.4
    p["bz"][urgent_unit] = 0.4 * 127 + 6 - energy_bias

    p["Wz"][full_unit] = energy
    p["bz"][full_unit] = energy_bias - 38

    p["Wz"][no_refill_unit, 56] = -0.1
    p["bz"][no_refill_unit] = 1

    # Refill mode is the only persistent phase flag.
    p["Wm"][6, 6] = 3
    p["Wmz"][6, urgent_unit] = 5
    p["Wmz"][6, full_unit] = -20
    p["Wmz"][6, no_refill_unit] = -300
    p["bm"][6] = -1

    zero = np.zeros(128, dtype=np.float32)

    # Current-observation phase eligibility:
    # shape first, then colour, then orientation, then goal.
    # Each tuple describes an affine "ineligible" indicator.
    phase_rules = (
        (shape_error + colour_error + pixel_error, 0.5, 0.5),
        (-shape_error, 1.5, 0.5),
        (shape_error - colour_error, 1.5, 0.5),
        (shape_error + colour_error - pixel_error, 1.5, 0.5),
        (zero, 0.5, -0.5),
    )

    for field, (error_vector, constant, refill_coefficient) in enumerate(
        phase_rules
    ):
        for a in range(4):
            unit = 8 + 4 * field + a
            p["Wmz"][unit] = (
                4 * nav_vectors[field][a] - 12 * error_vector
            )
            p["bm"][unit] = (
                4 * nav_constants[field][a] - 2 - 12 * constant
            )
            p["Wm"][unit, 6] = -12 * refill_coefficient
            p["Wo"][1 + a, 128 + unit] = 5

    # Stable tie-breaking: preserve heading, then prefer a right turn.
    right_turn = (3, 2, 0, 1)
    left_turn = (2, 3, 1, 0)
    for heading in range(4):
        unit = 28 + heading
        p["Wma"][unit, 1 + heading] = 5
        p["Wo"][1 + heading, 128 + unit] = 0.40
        p["Wo"][1 + right_turn[heading], 128 + unit] = 0.15
        p["Wo"][1 + left_turn[heading], 128 + unit] = 0.05

    # Strong penalties for visually blocked moves.
    for a in range(4):
        p["Wo"][1 + a, wall_plus[a]] = -100
        p["Wo"][1 + a, wall_minus[a]] = 100

    def coordinate_vector(a):
        result = np.zeros(128, dtype=np.float32)
        result[raw_units[a]] = 1
        result[wall_plus[a]] = -2
        result[wall_minus[a]] = 2
        return result

    x_position = coordinate_vector(0)
    y_position = coordinate_vector(1)
    p["Wmz"][32] = 0.1 * x_position
    p["Wmz"][33] = 0.1 * y_position

    # Remember failed movement attempts, including invisible constraints.
    for a in range(4):
        unit = 34 + a
        position = y_position if a < 2 else x_position
        previous_position = 33 if a < 2 else 32
        sign = -1 if a in (0, 2) else 1

        p["Wmz"][unit] = -100 * sign * position
        p["Wm"][unit, previous_position] = 1000 * sign
        p["Wma"][unit, 1 + a] = 200
        p["bm"][unit] = 3.9 - 200
        p["Wo"][1 + a, 128 + unit] = -12

    p["bo"][:] = -10000
    p["bo"][1:5] = 0

    return p
