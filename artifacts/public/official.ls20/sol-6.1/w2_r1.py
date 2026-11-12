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

    colours = [12, 9, 14, 8]
    bit_weights = (2.0 ** np.arange(9)).astype(np.float32)

    # Doubled UI glyph samples.
    for i in range(9):
        dy = 2 * (i // 3 - 1)
        dx = 2 * (i % 3 - 1)
        for colour in colours:
            p["W1"][i, colour, 2 + dy, 2 + dx] = 1.0

    # Black frame surrounding a goal glyph.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if abs(dy) == 2 or abs(dx) == 2:
                p["W1"][9, 5, 2 + dy, 2 + dx] = 1.0
    p["b1"][9] = -15.0

    # Coordinate hinges identifying the UI glyph anchor.
    for ch, axis, sign, anchor in (
        (10, 17, 1.0, 5),
        (11, 17, -1.0, 5),
        (12, 16, 1.0, 57),
        (13, 16, -1.0, 57),
    ):
        p["W1"][ch, axis, 2, 2] = sign
        p["b1"][ch] = -sign * anchor / 63.0

    # Exact player-centre detector.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            colour = 12 if dy < 0 else 9
            p["W1"][14, colour, 2 + dy, 2 + dx] = 1.0
    p["b1"][14] = -24.0

    # Shape transformer.
    for dy, dx in [(-1, -1), (0, 0), (0, 1), (1, 0)]:
        p["W1"][15, 0, 2 + dy, 2 + dx] = 1.0
    p["b1"][15] = -3.0

    # Colour transformer.
    colour_icon = [
        [9, 14, 14],
        [9, 0, 8],
        [12, 12, 8],
    ]
    for i in range(3):
        for j in range(3):
            p["W1"][16, colour_icon[i][j], 1 + i, 1 + j] = 1.0
    p["b1"][16] = -8.0

    # Rotation transformer.
    for dy, dx, colour in (
        (-1, 0, 0),
        (0, -1, 1),
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 1),
    ):
        p["W1"][17, colour, 2 + dy, 2 + dx] = 1.0
    p["b1"][17] = -4.0

    # Clearance count for a 5x5 player.
    p["W1"][18, 4, :, :] = 1.0

    # Raw glyph colours and colour-presence clipping hinges.
    for k, colour in enumerate(colours):
        p["W1"][19 + k, colour, 2, 2] = 1.0
        p["W1"][23 + k, colour, :, :] = 1.0
        p["W1"][27 + k, colour, :, :] = 1.0
        p["b1"][23 + k] = -4.0
        p["b1"][27 + k] = -5.0

    p["W1"][19, 11, 2, 2] = 1.0
    p["W1"][31, 11, 2, 2] = 1.0

    # Goal glyph bits.
    for i in range(9):
        dy = i // 3 - 1
        dx = i % 3 - 1
        for k in range(4):
            p["W2"][i, 19 + k, 1 + dy, 1 + dx] = 1.0
            p["W2"][i, 23 + k, 1, 1] = 1.0
            p["W2"][i, 27 + k, 1, 1] = -1.0
        p["W2"][i, 9, 1, 1] = 2.0
        p["b2"][i] = -3.0

    # UI bits, restricted to the canonical UI anchor.
    for i in range(9):
        p["W2"][9 + i, i, 1, 1] = 1.0
        for ch in range(10, 14):
            p["W2"][9 + i, ch, 1, 1] = -1000.0

    # Colour-9 goal presence.
    p["W2"][18, 24, 1, 1] = 1.0
    p["W2"][18, 28, 1, 1] = -1.0
    p["W2"][18, 9, 1, 1] = 1.0
    p["b2"][18] = -1.0

    # Other-goal colour index.
    for k in (2, 3):
        p["W2"][19, 23 + k, 1, 1] = float(k)
        p["W2"][19, 27 + k, 1, 1] = -float(k)
    p["W2"][19, 9, 1, 1] = 5.0
    p["b2"][19] = -5.0

    # UI colour index.
    for k in range(4):
        p["W2"][20, 23 + k, 1, 1] = float(k)
        p["W2"][20, 27 + k, 1, 1] = -float(k)
    for ch in range(10, 14):
        p["W2"][20, ch, 1, 1] = -1000.0

    for out_ch, in_ch in (
        (21, 14),
        (22, 15),
        (23, 16),
        (24, 17),
        (25, 18),
    ):
        p["W2"][out_ch, in_ch, 1, 1] = 1.0

    # Player markers: right, left, down, up.
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for k, (dy, dx) in enumerate(directions):
        p["W2"][26 + k, 14, 1 - dy, 1 - dx] = 1.0

    # Battery ring.
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy or dx:
                p["W2"][30, 31, 1 + dy, 1 + dx] = 1.0
    p["W2"][30, 31, 1, 1] = -8.0
    p["b2"][30] = -7.0

    # Blue energy-bar pixels.
    p["W2"][31, 31, 1, 1] = 1.0
    p["W2"][31, 13, 1, 1] = -1000.0
    p["W2"][31, 11, 1, 1] = -1000.0

    # Pointwise visual representation.
    for out_ch, in_ch in (
        (0, 21),
        (1, 22),
        (2, 23),
        (3, 24),
        (4, 25),
    ):
        p["W3"][out_ch, in_ch] = 1.0

    p["W3"][5, 9:18] = bit_weights

    p["W3"][6, 0:9] = bit_weights
    p["W3"][6, 18] = 512.0
    p["b3"][6] = -512.0

    p["W3"][7, 0:9] = bit_weights
    p["W3"][7, 18] = -512.0

    p["W3"][8, 18] = 1.0
    p["W3"][9, 0:9] = 1.0
    p["W3"][9, 18] = -9.0

    p["W3"][10, 20] = 1.0
    p["W3"][11, 19] = 1.0

    # Goal sources. Other colours ignite later than colour 9.
    p["W3"][12, 18] = 1.0
    p["W3"][13, 0:9] = 1.0e-12
    p["W3"][13, 18] = -9.0e-12

    for k in range(4):
        p["W3"][14 + k, 26 + k] = 1.0

    p["W3"][18, 30] = 1.0

    # Local movement vetoes.
    for k in range(4):
        p["W3"][19 + k, 25] = 1.0
        p["W3"][19 + k, 26 + k] = 100.0
        p["b3"][19 + k] = -100.0

    p["W3"][23, 31] = 1.0

    # Four recurrent spatial fields:
    # rotation, shape/battery, colour, and goals.
    #
    # A-B is clipped reachability. Two reader channels pack four
    # directional reachability integrals using distinct baselines.
    seed_features = [3, 1, 2, None]
    cardinal = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for field in range(4):
        base = 4 * field

        for out_ch in (base, base + 1):
            for dy, dx in cardinal:
                p["Wh"][out_ch, base, 1 + dy, 1 + dx] = 1.0
                p["Wh"][out_ch, base + 1, 1 + dy, 1 + dx] = -1.0
            p["Uh"][out_ch, 4] = -1000.0

            feature = seed_features[field]
            if feature is not None:
                p["Uh"][out_ch, feature] = 1.0
            else:
                p["Uh"][out_ch, 12] = 1.0
                p["Uh"][out_ch, 13] = 1.0

            if field == 1:
                p["Uh"][out_ch, 18] = 1.0e-32

        p["bh"][base + 1] = -1.0

        for reader, positive_marker, negative_marker in (
            (base + 2, 14, 15),
            (base + 3, 16, 17),
        ):
            p["Wh"][reader, reader, 1, 1] = 1.0
            p["Wh"][reader, base, 1, 1] = 1.0
            p["Wh"][reader, base + 1, 1, 1] = -1.0
            p["Uh"][reader, positive_marker] = 1004.0
            p["Uh"][reader, negative_marker] = 1008.0
            p["bh"][reader] = -1000.0

    # Analytical decoder for the six shapes at four rotations.
    masks = [
        [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
        [[0, 1, 0], [0, 1, 0], [1, 1, 1]],
        [[1, 0, 1], [1, 0, 1], [1, 1, 1]],
        [[0, 1, 1], [1, 0, 1], [0, 1, 0]],
        [[0, 1, 0], [1, 1, 0], [0, 1, 1]],
        [[1, 1, 1], [0, 0, 1], [1, 0, 1]],
    ]

    glyphs = []
    for shape_id, mask in enumerate(masks):
        mask = np.asarray(mask, dtype=np.int32)
        for rotation in range(4):
            rotated = np.rot90(mask, -rotation)
            code = int(np.dot(rotated.ravel(), bit_weights))
            glyphs.append((code, shape_id, rotation))
    glyphs.sort()

    xs = np.asarray([0] + [g[0] for g in glyphs], dtype=np.float64)
    shape_values = np.asarray(
        [0] + [g[1] for g in glyphs], dtype=np.float64
    )
    rotation_values = np.asarray(
        [0] + [g[2] for g in glyphs], dtype=np.float64
    )

    def coefficients(values):
        slopes = np.diff(values) / np.diff(xs)
        return np.concatenate(
            (slopes[:1], np.diff(slopes))
        ).astype(np.float32)

    shape_coeff = coefficients(shape_values)
    rotation_coeff = coefficients(rotation_values)

    # g layout: h.max, h.mean, f3.max, f3.mean.
    cursor = 0
    code_bases = {}

    for label, feature in (("ui", 5), ("red", 6), ("other", 7)):
        code_bases[label] = cursor
        for knot in xs[:-1]:
            p["Wz"][cursor, 32 + feature] = 1.0
            p["bz"][cursor] = -float(knot)
            cursor += 1

    # Recover right, left, down, and up arrival-time scores.
    navigation_units = {}
    for field in range(4):
        base = 4 * field
        units = []

        for reader, positive in (
            (base + 2, True),
            (base + 2, False),
            (base + 3, True),
            (base + 3, False),
        ):
            unit = cursor
            if positive:
                p["Wz"][unit, 16 + reader] = 4096.0
                p["Wz"][unit, reader] = -1.0
                p["bz"][unit] = -512.0
            else:
                p["Wz"][unit, reader] = 1.0
                p["bz"][unit] = -1024.0
            units.append(unit)
            cursor += 1

        navigation_units[field] = units

    def feature_unit(feature, mean=False, scale=1.0):
        nonlocal cursor
        unit = cursor
        offset = 64 if mean else 32
        p["Wz"][unit, offset + feature] = scale
        cursor += 1
        return unit

    ui_colour_unit = feature_unit(10)
    other_colour_unit = feature_unit(11)
    red_exists_unit = feature_unit(8)
    other_exists_unit = feature_unit(9)
    illegal_units = [feature_unit(19 + k) for k in range(4)]
    battery_unit = feature_unit(18)
    shape_presence_unit = feature_unit(1)
    energy_unit = feature_unit(23, mean=True, scale=2048.0)

    # Two signed comparison flags per attribute and goal queue.
    for group, label in enumerate(("red", "other")):
        for attribute in range(3):
            for sign_index, sign in enumerate((1.0, -1.0)):
                mem_ch = 6 * group + 2 * attribute + sign_index
                scale = 20.0 * sign
                p["bm"][mem_ch] = -10.0

                if attribute in (0, 2):
                    coeff = (
                        shape_coeff if attribute == 0
                        else rotation_coeff
                    )
                    ui_base = code_bases["ui"]
                    goal_base = code_bases[label]
                    p["Wmz"][mem_ch, ui_base:ui_base + 24] += scale * coeff
                    p["Wmz"][mem_ch, goal_base:goal_base + 24] -= scale * coeff
                else:
                    p["Wmz"][mem_ch, ui_colour_unit] = scale
                    if label == "red":
                        p["bm"][mem_ch] -= scale
                    else:
                        p["Wmz"][mem_ch, other_colour_unit] = -scale

    p["Wmz"][12, red_exists_unit] = 20.0
    p["bm"][12] = -10.0
    p["Wmz"][13, other_exists_unit] = 20.0
    p["bm"][13] = -10.0

    constant_mem = 54
    p["bm"][constant_mem] = 20.0

    # Battery mode when energy is low and the shape field is free.
    power_mem = 59
    p["Wmz"][power_mem, energy_unit] = -20.0
    p["Wmz"][power_mem, battery_unit] = 10000.0
    p["Wmz"][power_mem, shape_presence_unit] = -10000.0
    p["bm"][power_mem] = 400.0 - 10000.0

    action_for_direction = [4, 3, 2, 1]

    # Attribute priority: shape, colour, rotation, then goal.
    for group in range(2):
        flag_base = 6 * group

        for stage in range(4):
            first_mem = 14 + 5 * (4 * group + stage)
            expr = np.zeros(64, dtype=np.float32)

            if stage == 0:
                expr[flag_base:flag_base + 2] = -1.0
                constant = 0.0
                field = 1
            elif stage == 1:
                expr[flag_base:flag_base + 2] = 1.0
                expr[flag_base + 2:flag_base + 4] = -1.0
                constant = 2.0
                field = 2
            elif stage == 2:
                expr[flag_base:flag_base + 4] = 1.0
                expr[flag_base + 4:flag_base + 6] = -1.0
                constant = 4.0
                field = 0
            else:
                expr[flag_base:flag_base + 6] = 1.0
                constant = 6.0
                field = 3

            if group == 0:
                expr[12] -= 1.0
                constant += 1.0
            else:
                expr[12] += 1.0
                expr[13] -= 1.0
                constant += 2.0

            expr[power_mem] += 1.0
            constant += 1.0
            expr[constant_mem] += constant

            penalty = -20.0 * expr
            baseline_mem = first_mem + 4

            for mem_ch in range(first_mem, first_mem + 5):
                p["Wm"][mem_ch] = penalty

            for direction, unit in enumerate(navigation_units[field]):
                mem_ch = first_mem + direction
                p["Wmz"][mem_ch, unit] = 0.002
                action = action_for_direction[direction]
                p["Wo"][action, 128 + mem_ch] += 20.0
                p["Wo"][action, 128 + baseline_mem] -= 20.0

    # Battery navigation.
    for direction, unit in enumerate(navigation_units[1]):
        mem_ch = 60 + direction
        p["Wm"][mem_ch, power_mem] = 20.0
        p["bm"][mem_ch] = -20.0
        p["Wmz"][mem_ch, unit] = 0.002
        action = action_for_direction[direction]
        p["Wo"][action, 128 + mem_ch] = 20.0

    # Immediate weak goal navigation breaks the controller's
    # one-step delay when a transformer is hidden by the player.
    for direction, unit in enumerate(navigation_units[3]):
        action = action_for_direction[direction]
        p["Wo"][action, unit] += 0.00002

    # Strong local wall vetoes.
    for direction, unit in enumerate(illegal_units):
        p["Wo"][action_for_direction[direction], unit] = -1000.0

    # Very weak persistence, below meaningful wave differences.
    for k, action in enumerate((1, 2, 3, 4)):
        mem_ch = 55 + k
        p["Wma"][mem_ch, action] = 4.0
        p["bm"][mem_ch] = -4.0
        p["Wo"][action, 128 + mem_ch] = 0.000001

    p["bo"][[0, 5, 6, 7]] = -1000.0
    return p
