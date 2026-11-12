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

    anchors = [
        (24, 30), (35, 45), (35, 46), (40, 43),
        (53, 30), (53, 22), (35, 17), (35, 18),
    ]
    probes = [(3, 4), (4, 3), (4, 12), (12, 5), (3, 6)]
    brush_point = (20, 31)

    rows = sorted({r for r, _ in anchors + probes + [brush_point]})
    cols = sorted({c for _, c in anchors + probes + [brush_point]})

    ch = 0
    row_base = ch
    ch += 1
    p["W1"][row_base, 16, 2, 2] = 63.0

    col_base = ch
    ch += 1
    p["W1"][col_base, 17, 2, 2] = 63.0

    row_hinge = {}
    for r in rows:
        row_hinge[r] = ch
        p["W1"][ch, 16, 2, 2] = 63.0
        p["b1"][ch] = -float(r)
        ch += 1

    col_hinge = {}
    for c in cols:
        col_hinge[c] = ch
        p["W1"][ch, 17, 2, 2] = 63.0
        p["b1"][ch] = -float(c)
        ch += 1

    red = ch
    ch += 1
    p["W1"][red, 2, 2, 2] = 1.0

    colour_code = ch
    ch += 1
    for colour in range(16):
        p["W1"][colour_code, colour, 2, 2] = float(colour + 1)

    swatch = ch
    ch += 1
    for dr in range(5):
        for dc in range(5):
            if dr in (0, 4) or dc in (0, 4):
                p["W1"][swatch, 4, dr, dc] = 1.0
    p["W1"][swatch, 4, 2, 2] = -1.0
    p["b1"][swatch] = -15.5

    brush = ch
    ch += 1
    p["W1"][brush, 2, 0, 2] = 1.0
    p["W1"][brush, 2, 1, 0] = 1.0
    p["W1"][brush, 2, 2, 0] = 1.0
    p["b1"][brush] = -2.5

    # Rectangle hinges and nonblack indicator for the upper canvas.
    extra = {}
    for axis, value in ((16, 34), (16, 38), (17, 27), (17, 36)):
        extra[(axis, value)] = ch
        p["W1"][ch, axis, 2, 2] = 63.0
        p["b1"][ch] = -float(value)
        ch += 1

    nonblack = ch
    ch += 1
    p["W1"][nonblack, 18, 2, 2] = 1.0
    p["W1"][nonblack, 0, 2, 2] = -1.0

    assert ch <= 32

    def position_gate(out_ch, source_ch, r, c, magnitude=1.0,
                      source_scale=1.0):
        w = p["W2"][out_ch]
        w[source_ch, 1, 1] += source_scale
        w[row_base, 1, 1] += magnitude
        w[col_base, 1, 1] += magnitude
        w[row_hinge[r], 1, 1] -= 2.0 * magnitude
        w[col_hinge[c], 1, 1] -= 2.0 * magnitude
        p["b2"][out_ch] -= magnitude * float(r + c)

    for i, (r, c) in enumerate(anchors):
        position_gate(i, red, r, c)

    # Disambiguate the adjacent diagonal and horizontal basket samples.
    p["W2"][2, red, 1, 0] -= 1.0

    for i, (r, c) in enumerate(probes):
        position_gate(8 + i, colour_code, r, c, magnitude=32.0)

    palette_mask = 13
    palette_code = 14
    brush_mask = 15
    p["W2"][palette_mask, swatch, 1, 1] = 2.0
    p["W2"][palette_code, swatch, 1, 1] = 64.0
    p["W2"][palette_code, colour_code, 1, 1] = 1.0
    p["b2"][palette_code] = -32.0
    position_gate(brush_mask, brush, *brush_point, source_scale=2.0)

    # Positive exactly at nonblack pixels in rows 34..38, columns 27..36.
    dirty = 16
    p["W2"][dirty, nonblack, 1, 1] = 1.0
    p["W2"][dirty, row_base, 1, 1] = 1.0
    p["W2"][dirty, col_base, 1, 1] = 1.0
    for key in ((16, 34), (16, 38), (17, 27), (17, 36)):
        p["W2"][dirty, extra[key], 1, 1] = -1.0
    p["b2"][dirty] = -61.0

    # f3: basket indicators, vertically broadcast target samples,
    # palette and brush features, and basket-position groups.
    for i in range(8):
        p["W3"][i, i] = 1.0

    for i in range(5):
        p["W3"][8 + i, 64 + 8 + i] = 1.0

    p["W3"][13, palette_mask] = 1.0
    p["W3"][14, palette_code] = 1.0
    p["W3"][15, brush_mask] = 1.0
    p["W3"][21, dirty] = 1.0

    groups = [[6, 7], [5], [2, 3], [4], [1]]
    for group, positions in enumerate(groups):
        for pos in positions:
            p["W3"][16 + group, 64 + pos] = 1.0

    # Carry basket-group indicators left, then the selected colour right.
    for i in range(5):
        p["Uh"][i, 16 + i] = 1.0
        p["Wh"][i, i, 1, 2] = 1.0

    for i in range(5):
        p["Uh"][5 + i, 8 + i] = 1.0
        p["Wh"][5 + i, i, 1, 1] = 32.0
        p["bh"][5 + i] = -32.0
        p["Wh"][10, 5 + i, 1, 1] = 1.0
    p["Wh"][10, 10, 1, 0] = 1.0

    # Palette-colour absolute difference.
    p["Wh"][11, 10, 1, 1] = 1.0
    p["Uh"][11, 14] = -1.0
    p["Wh"][12, 10, 1, 1] = -1.0
    p["Uh"][12, 14] = 1.0

    p["Wc"][0, 13] = 32.0
    p["Wc"][0, 15] = 64.0
    p["Wc"][0, 32 + 11] = -1.0
    p["Wc"][0, 32 + 12] = -1.0

    for i in range(8):
        p["Wz"][i, 32 + i] = 1.0

    # Features for a clipped difference between two target samples.
    p["Wz"][8, 32 + 8] = 1.0
    p["Wz"][8, 32 + 9] = -1.0
    p["Wz"][9, 32 + 9] = 1.0
    p["Wz"][9, 32 + 8] = -1.0
    p["Wz"][10] = p["Wz"][8]
    p["Wz"][11] = p["Wz"][9]
    p["bz"][10] = -1.0
    p["bz"][11] = -1.0

    # Memory 0 counts broad strokes; memory 1 tracks palette selection.
    p["Wm"][0, 0] = 1.0
    p["Wma"][0, 5] = 0.01

    p["Wm"][1, 1] = 5.0
    p["Wma"][1, 6] = 10.0
    p["Wma"][1, 5] = -10.0
    p["Wma"][1, 0] = -20.0
    p["bm"][1] = -2.0

    p["Wo"][5, 128 + 1] = 0.75
    p["bo"][5] = 0.75

    clockwise = [4, 2, 2, 3, 3, 1, 1, 4]
    opposite = {1: 2, 2: 1, 3: 4, 4: 3}

    def route(pos, destination):
        if pos == destination:
            return 6
        forward = (destination - pos) % 8
        backward = (pos - destination) % 8
        if forward <= backward:
            return clockwise[pos]
        return opposite[clockwise[(pos - 1) % 8]]

    def score_vector(action):
        result = np.zeros(8, dtype=np.float32)
        result[action] = 1.0 if action == 6 else 2.0
        return result

    recipe_a = [4, 2, 7, 1]
    recipe_b = [6, 5, 3, 1]

    for pos in range(8):
        p["Wo"][:, pos] += score_vector(route(pos, recipe_a[0]))

    next_mem = 2

    def add_class_weight(mem_index, weight):
        p["Wmz"][mem_index, 8] += weight
        p["Wmz"][mem_index, 9] += weight
        p["Wmz"][mem_index, 10] -= weight
        p["Wmz"][mem_index, 11] -= weight

    def add_gate(pos, difference, kind, threshold=None):
        nonlocal next_mem
        difference = np.asarray(difference, dtype=np.float32)
        if not np.any(difference):
            return None

        idx = next_mem
        next_mem += 1
        scale = 2.0

        if kind == "stage":
            p["Wmz"][idx, pos] = scale * 100.0
            p["Wm"][idx, 0] = scale * 1000.0
            p["Wma"][idx, 5] = scale * 10.0
            p["bm"][idx] = scale * (-100.0 - 1000.0 * threshold)

        elif kind == "class":
            p["Wmz"][idx, pos] = scale * 100.0
            add_class_weight(idx, scale * 100.0)
            p["bm"][idx] = scale * -150.0

        elif kind == "class_stage":
            p["Wmz"][idx, pos] = scale * 100.0
            add_class_weight(idx, scale * 100.0)
            p["Wm"][idx, 0] = scale * 1000.0
            p["Wma"][idx, 5] = scale * 10.0
            p["bm"][idx] = scale * (-200.0 - 1000.0 * threshold)

        elif kind == "class_first_clicked":
            p["Wmz"][idx, pos] = scale * 100.0
            add_class_weight(idx, scale * 100.0)
            p["Wm"][idx, 1] = scale * 50.0
            p["Wm"][idx, 0] = scale * -10000.0
            p["Wma"][idx, 6] = scale * 100.0
            p["Wma"][idx, 5] = scale * -200.0
            p["bm"][idx] = scale * -225.0

        elif kind == "accent_clicked":
            p["Wmz"][idx, pos] = scale * 1000.0
            p["Wm"][idx, 1] = scale * 500.0
            p["Wm"][idx, 0] = scale * 10000.0
            p["Wma"][idx, 6] = scale * 1000.0
            p["Wma"][idx, 5] = scale * -900.0
            p["bm"][idx] = scale * -1750.0

        else:
            raise ValueError(kind)

        p["Wo"][:, 128 + idx] += 0.5 * difference
        p["bo"][:] += 0.5 * difference
        return idx

    thresholds = [0.005, 0.015, 0.025]

    for stage in range(1, 4):
        for pos in range(8):
            before = score_vector(route(pos, recipe_a[stage - 1]))
            after = score_vector(route(pos, recipe_a[stage]))
            add_gate(pos, after - before, "stage", thresholds[stage - 1])

    for pos in range(8):
        a0 = score_vector(route(pos, recipe_a[0]))
        b0 = score_vector(route(pos, recipe_b[0]))
        add_gate(pos, b0 - a0, "class")

    for stage in range(1, 4):
        for pos in range(8):
            old_correction = (
                score_vector(route(pos, recipe_b[stage - 1]))
                - score_vector(route(pos, recipe_a[stage - 1]))
            )
            new_correction = (
                score_vector(route(pos, recipe_b[stage]))
                - score_vector(route(pos, recipe_a[stage]))
            )
            add_gate(
                pos, new_correction - old_correction,
                "class_stage", thresholds[stage - 1]
            )

    for pos in range(8):
        before = score_vector(route(pos, 6))
        after = score_vector(route(pos, 0))
        add_gate(pos, after - before, "class_first_clicked")

    accent_zero_gate = None
    for pos in range(8):
        before = score_vector(route(pos, 1))
        after = score_vector(route(pos, 0))
        idx = add_gate(pos, after - before, "accent_clicked")
        if pos == 0:
            accent_zero_gate = idx

    if accent_zero_gate is not None:
        p["Wo"][6, 128 + accent_zero_gate] += 1.0
        p["bo"][6] += 1.0

    # Special path for a black upper half and white lower half.
    # Target sample colour codes are colour + 1.
    special_z = 12
    for pos in range(8):
        z = special_z + pos
        p["Wz"][z, 32 + 11] = 2.0  # bottom sample
        p["Wz"][z, 32 + 8] = -2.0  # top sample
        p["Wz"][z, 32 + 9] = -2.0  # left sample
        p["Wz"][z, 32 + pos] = 2.0
        p["bz"][z] = -29.5
        p["Wo"][route(pos, 4), z] += 40.0

    # Only inspect the upper canvas for contamination when the basket
    # is at the bottom, where it cannot cover the inspected region.
    reset_z = 20
    p["Wz"][reset_z, 32 + 11] = 2.0
    p["Wz"][reset_z, 32 + 8] = -2.0
    p["Wz"][reset_z, 32 + 9] = -2.0
    p["Wz"][reset_z, 32 + 4] = 2.0
    p["Wz"][reset_z, 32 + 21] = 2.0
    p["bz"][reset_z] = -30.5
    p["Wo"][0, reset_z] = 500.0

    # After the special path clicks the white swatch, paint immediately.
    special_clicked = next_mem
    next_mem += 1
    p["Wmz"][special_clicked, special_z + 4] = 20.0
    p["Wm"][special_clicked, 1] = 10.0
    p["Wma"][special_clicked, 6] = 20.0
    p["bm"][special_clicked] = -15.0
    p["Wo"][5, 128 + special_clicked] += 30.0
    p["Wo"][6, 128 + special_clicked] -= 30.0
    p["bo"][5] += 30.0
    p["bo"][6] -= 30.0

    assert next_mem <= 64
    return p
