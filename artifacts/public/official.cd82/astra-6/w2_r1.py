import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,),
        "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,),
        "Wh": (16, 16, 3, 3),
        "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,),
        "Wm": (64, 64), "Wmz": (64, 128),
        "Wma": (64, 8), "bm": (64,),
        "Wo": (8, 192), "bo": (8,),
        "Wc": (1, 48), "bc": (1,),
    }
    p = {k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()}

    count = 0

    def new_channel():
        nonlocal count
        c = count
        count += 1
        return c

    def first(o, i, dy=0, dx=0, v=1.0):
        p["W1"][o, i, dy + 2, dx + 2] += v

    def second(o, i, dy=0, dx=0, v=1.0):
        p["W2"][o, i, dy + 1, dx + 1] += v

    def recurrent(o, i, dy=0, dx=0, v=1.0):
        p["Wh"][o, i, dy + 1, dx + 1] += v

    def hinge(x, y):
        c = new_channel()
        first(c, 16, v=4032.0)
        first(c, 17, v=63.0)
        p["b1"][c] = -(64 * y + x)
        return c

    # Coordinate hinges whose second differences select individual pixels.
    target_anchors = {x: hinge(3 + x, 7) for x in (3, 4, 5, 6)}
    canvas_anchors = {x: hinge(27 + x, 38) for x in (3, 4, 5, 6)}

    basket_points = [
        (31, 25), (39, 24), (45, 38), (46, 46),
        (31, 52), (24, 53), (18, 38), (17, 31),
    ]
    basket_hinges = [hinge(x, y) for x, y in basket_points]

    # Each detail sample corresponds to the first clockwise core wedge
    # of that cardinal direction.
    patch_points = [(4, 1), (8, 4), (5, 8), (1, 5)]
    patch_hinges = [hinge(3 + x, 3 + y) for x, y in patch_points]

    raw = new_channel()
    for colour in range(16):
        first(raw, colour, v=float(colour + 1))

    bad = new_channel()
    first(bad, 2)
    first(bad, 5)

    palette_pattern = new_channel()
    for dy, dx in ((-2, 0), (0, -2), (0, 2)):
        first(palette_pattern, 4, dy, dx)
    p["b1"][palette_pattern] = -2.0

    underline = new_channel()
    first(underline, 0, 2, 0)

    # The small tool has two black pixels bracketed by red pixels.
    tool_horizontal = new_channel()
    first(tool_horizontal, 0)
    first(tool_horizontal, 0, 0, 1)
    first(tool_horizontal, 2, 0, -1)
    first(tool_horizontal, 2, 0, 2)
    p["b1"][tool_horizontal] = -3.0

    tool_vertical = new_channel()
    first(tool_vertical, 0)
    first(tool_vertical, 0, 1, 0)
    first(tool_vertical, 2, -1, 0)
    first(tool_vertical, 2, 2, 0)
    p["b1"][tool_vertical] = -3.0

    def add_hat(out, source, dy=0, scale=1.0):
        second(out, source, dy, -1, scale)
        second(out, source, dy, 0, -2.0 * scale)
        second(out, source, dy, 1, scale)

    gate = 20.0
    sectors = [
        (4, 3), (5, 3), (6, 4), (6, 5),
        (5, 6), (4, 6), (3, 5), (3, 4),
    ]

    def sector_feature(out, sector, anchors):
        x, y = sectors[sector]
        output_y = 5 if y == 6 else 4
        add_hat(out, anchors[x], 4 - output_y, gate)
        second(out, raw, y - output_y, 0)
        p["b2"][out] = -gate

    # Target and current colours in the trailing wedge of every basket.
    for i in range(8):
        s = (i + 7) % 8
        sector_feature(i, s, target_anchors)
        sector_feature(8 + i, s, canvas_anchors)

    # Basket-presence indicators.
    for i, source in enumerate(basket_hinges):
        add_hat(16 + i, source)
        second(16 + i, bad, v=-1.0)

    # Target detail colour; deliberately use a consistent side,
    # rather than the minimum of two potentially different colours.
    for j in range(4):
        add_hat(24 + j, patch_hinges[j], -1, gate)
        second(24 + j, raw, -1, 0)
        p["b2"][24 + j] = -gate

    second(28, raw)
    second(28, palette_pattern, v=gate)
    p["b2"][28] = -gate

    second(29, raw)
    second(29, palette_pattern, v=gate)
    second(29, underline, 1, 0, gate)
    p["b2"][29] = -2.0 * gate

    second(30, tool_horizontal)
    second(31, tool_vertical)

    ROW, COL = 32, 64

    # Axial intersections select values belonging to the active basket.
    for i in range(8):
        p["W3"][i, ROW + i] = 1.0
        p["W3"][i, COL + 16 + i] = gate
        p["b3"][i] = -gate

        p["W3"][8 + i, ROW + 8 + i] = 1.0
        p["W3"][8 + i, COL + 16 + i] = gate
        p["b3"][8 + i] = -gate

    for j, orientation in enumerate((0, 2, 4, 6)):
        out = 16 + j
        if orientation in (0, 4):
            p["W3"][out, ROW + 24 + j] = 1.0
            p["W3"][out, COL + 16 + orientation] = gate
            p["W3"][out, COL + 30] = gate
        else:
            p["W3"][out, COL + 24 + j] = 1.0
            p["W3"][out, ROW + 16 + orientation] = gate
            p["W3"][out, ROW + 31] = gate
        p["b3"][out] = -2.0 * gate

        # Matching inner target wedge, used to detect a distinct detail.
        core_source = 2 * j + 1
        p["W3"][24 + j, ROW + core_source] = 1.0
        p["W3"][24 + j, COL + 16 + orientation] = gate
        p["b3"][24 + j] = -gate

    p["W3"][20, 28] = 1.0
    p["W3"][21, 29] = 1.0
    p["W3"][22, 30] = 1.0
    p["W3"][22, 31] = 1.0

    # Complement of the valid click-location mask. This prevents
    # unfinished spatial propagation from selecting empty screen cells.
    p["b3"][23] = 1.0
    p["W3"][23, 28] = -2.0
    p["W3"][23, 30] = -2.0
    p["W3"][23, 31] = -2.0

    # Directed prefix/broadcast circuits. Values travel to a boundary,
    # upward, and then across the palette row.
    def route(base, sources, right_boundary):
        for source in sources:
            p["Uh"][base, source] = 1.0

        if right_boundary:
            recurrent(base, base, 0, -1)
            recurrent(base + 1, base + 1, 1, 0)
            recurrent(base + 1, base)
            recurrent(base + 1, base, 0, 1, -1.0)
            recurrent(base + 2, base + 2, 0, 1)
        else:
            recurrent(base, base, 0, 1)
            recurrent(base + 1, base + 1, 1, 0)
            recurrent(base + 1, base)
            recurrent(base + 1, base, 0, -1, -1.0)
            recurrent(base + 2, base + 2, 0, -1)

        recurrent(base + 2, base + 1)

    route(0, range(8), True)
    route(3, range(8, 16), True)
    route(6, range(16, 20), False)

    # Compare target and current trailing-wedge colours.
    recurrent(9, 2)
    recurrent(9, 5, v=-1.0)
    recurrent(10, 5)
    recurrent(10, 2, v=-1.0)

    # Flat-topped equality indicator, robust to coordinate roundoff.
    for out, bias in ((11, 0.75), (12, 0.25)):
        recurrent(out, 9, v=-1.0)
        recurrent(out, 10, v=-1.0)
        p["bh"][out] = bias

    # Choose main colour when the wedge is wrong; detail colour otherwise.
    recurrent(13, 8)
    recurrent(13, 11, v=40.0)
    recurrent(13, 12, v=-40.0)
    p["bh"][13] = -20.0

    recurrent(14, 2)
    recurrent(14, 11, v=-40.0)
    recurrent(14, 12, v=40.0)

    recurrent(15, 13)
    recurrent(15, 14)
    p["Uh"][15, 20] = -1.0

    # Palette score is a colour-equality score. Suppressing the selected
    # swatch makes the tool win once its requested colour is selected.
    p["Wc"][0, 20] = -1900.0
    p["Wc"][0, 21] = -10000.0
    p["Wc"][0, 22] = 10.0
    p["Wc"][0, 23] = -1000000.0
    p["Wc"][0, 32 + 13] = 2000.0
    p["Wc"][0, 32 + 14] = 2000.0
    p["Wc"][0, 32 + 15] = -4000.0

    # Global scalar readouts.
    D, C, B = 0, 3, 6
    ACTIVE = 32 + 21

    for i in range(8):
        p["Wz"][i, 32 + i] = 1.0

    def difference_pair(z, lhs, rhs):
        p["Wz"][z, lhs] = 1.0
        p["Wz"][z, rhs] = -1.0
        p["Wz"][z + 1, rhs] = 1.0
        p["Wz"][z + 1, lhs] = -1.0

    difference_pair(8, D, ACTIVE)
    difference_pair(10, D, C)
    difference_pair(12, B, ACTIVE)
    p["Wz"][14, 32 + 22] = 1.0

    # Difference between a detail sample and its corresponding core.
    p["Wz"][15, B] = 1.0
    p["Wz"][16, B] = -1.0
    for j in range(4):
        p["Wz"][15, 32 + 24 + j] = -1.0
        p["Wz"][16, 32 + 24 + j] = 1.0

    sharp = 20.0

    def equality_bit(index, terms):
        p["bm"][index] = sharp * 0.20
        for z in terms:
            p["Wmz"][index, z] = -sharp

    # Memory bits:
    # 0 main colour selected; 1 wedge correct; 2 tool click ready;
    # 3 preceding action completed a ready tool click;
    # 5 distinct detail required; 8:16 active orientation.
    equality_bit(0, (8, 9))
    equality_bit(1, (10, 11))
    equality_bit(2, (10, 11, 12, 13))

    p["Wm"][3, 2] = sharp
    p["Wma"][3, 6] = sharp
    p["bm"][3] = -1.5 * sharp

    p["Wmz"][5, 15] = sharp
    p["Wmz"][5, 16] = sharp
    p["Wmz"][5, 14] = 400.0
    p["bm"][5] = -400.0 - 0.20 * sharp

    for i in range(8):
        p["Wmz"][8 + i, i] = sharp
        p["bm"][8 + i] = -0.5 * sharp

    p["bo"][:] = -1000.0
    p["bo"][6] = 0.0

    def memory_output(action, index, weight):
        p["Wo"][action, 128 + index] += weight

    # Execute a large stroke only when needed and correctly coloured.
    p["bo"][5] = -4.0
    memory_output(5, 0, 1.0)
    memory_output(5, 1, -1.5)
    memory_output(5, 3, -2.5)

    # Clockwise movement around the eight legal basket positions.
    move_groups = {
        1: (5, 6),
        2: (1, 2),
        3: (3, 4),
        4: (0, 7),
    }
    for action, orientations in move_groups.items():
        p["bo"][action] = 0.5
        memory_output(action, 3, 1.5)
        memory_output(action, 1, 1.0)
        memory_output(action, 5, -1.0)
        for orientation in orientations:
            memory_output(action, 8 + orientation, 50.0)

    return p
