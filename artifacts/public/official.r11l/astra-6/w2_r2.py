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

    def c1(out, inp, dy=0, dx=0, value=1.0):
        p["W1"][out, inp, dy + 2, dx + 2] += value

    def c2(out, inp, dy=0, dx=0, value=1.0):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    def recurrent(out, inp, dy=0, dx=0, value=1.0):
        p["Wh"][out, inp, dy + 1, dx + 1] += value

    def feature(out, inp, reduction="local", value=1.0):
        offset = {"local": 0, "row": 32, "col": 64}[reduction]
        p["W3"][out, offset + inp] += value

    colours = (15, 14, 8, 9, 11)
    diamond = [
        (dy, dx)
        for dy in range(-2, 3)
        for dx in range(-2, 3)
        if abs(dy) + abs(dx) <= 2
    ]
    disk = [
        (dy, dx)
        for dy in range(-2, 3)
        for dx in range(-2, 3)
        if not (abs(dy) == 2 and abs(dx) == 2)
    ]

    # First layer:
    # 0:5 selected-family anchor centres
    # 5   selection-independent anchor centre
    # 6   colour-15 ball centre
    # 7:27 goal quadrants
    # 27:29 coordinates
    # 29:31 click exclusion
    # 31  anchor footprint against walls
    for family, colour in enumerate(colours):
        for dy, dx in diamond:
            if dy == 0 and dx == 0:
                c1(family, colour)
            else:
                c1(family, 0, dy, dx)
        p["b1"][family] = -12.0

    for dy, dx in diamond:
        if dy == 0 and dx == 0:
            for colour in colours:
                c1(5, colour)
        else:
            c1(5, 0, dy, dx)
            c1(5, 3, dy, dx)
    p["b1"][5] = -12.0

    for dy, dx in disk:
        colour = 6 if (dy == 0 and dx == 0) else 15
        c1(6, colour, dy, dx)
    p["b1"][6] = -20.0

    quadrant_centres = [
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]
    quadrant_points = [
        [(-3, -1), (-2, -2), (-1, -3)],
        [(-3, 1), (-2, 2), (-1, 3)],
        [(1, -3), (2, -2), (3, -1)],
        [(1, 3), (2, 2), (3, 1)],
    ]

    for family, colour in enumerate(colours):
        for q, ((cy, cx), points) in enumerate(
            zip(quadrant_centres, quadrant_points)
        ):
            out = 7 + 4 * family + q
            for dy, dx in points:
                c1(out, colour, dy - cy, dx - cx)
                if q < 2:
                    c1(out, 0, dy - cy, dx - cx)
                    c1(out, 3, dy - cy, dx - cx)

    c1(27, 17)
    c1(28, 16)

    for dy in range(-1, 2):
        for dx in range(-1, 2):
            for colour in (0, 3):
                c1(29, colour, dy, dx)
                c1(30, colour, dy, dx)
    p["b1"][30] = -1.0

    for dy, dx in diamond:
        c1(31, 2, dy, dx)

    # Second layer:
    # 0:5 selected-family markers
    # 5:7 selected coordinates
    # 7:9 selected-15 coordinates
    # 9:11 inverse ball coordinates
    # 11:21 goal coordinate moments
    # 21:26 goal markers
    # 26 inactive anchors
    # 27:29 coordinates
    # 29 click exclusion
    # 30 walls
    for family in range(5):
        c2(family, family)

    for out, coord in ((5, 27), (6, 28)):
        c2(out, coord)
        for family in range(5):
            c2(out, family)
        p["b2"][out] = -1.0

    for out, coord in ((7, 27), (8, 28)):
        c2(out, coord)
        c2(out, 0)
        p["b2"][out] = -1.0

    c2(9, 6)
    c2(9, 27, value=-1.0)
    c2(10, 6)
    c2(10, 28, value=-1.0)

    def ring_sum(out, family):
        for q, (dy, dx) in enumerate(quadrant_centres):
            c2(out, 7 + 4 * family + q, dy, dx)

    for family in range(5):
        for axis, coord in enumerate((27, 28)):
            out = 11 + 2 * family + axis
            ring_sum(out, family)
            c2(out, coord)
            p["b2"][out] = -12.0

        ring_sum(21 + family, family)
        p["b2"][21 + family] = -11.0

    c2(26, 5)
    for family in range(5):
        c2(26, family, value=-1.0)

    c2(27, 27)
    c2(28, 28)
    c2(29, 29)
    c2(29, 30, value=-1.0)
    c2(30, 31)

    # Axial gathering places each selected family's sources on
    # its selected anchor's row.
    feature(0, 27)
    feature(1, 28)
    feature(2, 7)
    feature(3, 8)

    for family in range(1, 5):
        feature(4, family)

    def row_gate(out, moment, family):
        feature(out, moment, "col")
        feature(out, family, "row")
        p["b3"][out] = -1.0

    row_gate(5, 11, 0)
    row_gate(6, 12, 0)
    row_gate(7, 9, 0)
    row_gate(8, 10, 0)

    row_gate(9, 13, 1)
    row_gate(10, 14, 1)

    feature(11, 5, "row")
    feature(12, 6, "row")
    for family in range(5):
        feature(13, family, "row")
    feature(14, 26)

    row_gate(15, 21, 0)
    row_gate(16, 22, 1)
    feature(17, 29)
    feature(18, 30)

    row_gate(19, 15, 2)
    row_gate(20, 16, 2)
    row_gate(21, 23, 2)

    row_gate(22, 17, 3)
    row_gate(23, 18, 3)
    row_gate(24, 24, 3)

    row_gate(25, 19, 4)
    row_gate(26, 20, 4)
    row_gate(27, 25, 4)

    presence_offset = 8.0
    x_offset = 2.0 + presence_offset
    y_offset = 2.0

    sx = np.zeros(32, dtype=np.float32)
    sy = np.zeros(32, dtype=np.float32)

    # Colour-15 feedback:
    # selected coordinate + 2 * (goal coordinate - ball coordinate).
    sx[2] = 1.0
    sx[5] = 2.0
    sx[7] = 2.0
    sy[3] = 1.0
    sy[6] = 2.0
    sy[8] = 2.0

    # Other monochromatic families cluster anchors above their goal.
    sx[4] = 2.0
    sy[4] = 2.0 - 4.0 / 63.0

    for x_channel, y_channel in (
        (9, 10), (19, 20), (22, 23), (25, 26)
    ):
        sx[x_channel] = 1.0
        sy[y_channel] = 1.0

    for presence_channel in (15, 16, 21, 24, 27):
        sx[presence_channel] = presence_offset

    # Horizontal inclusive prefix sums.
    for out, source, dx in (
        (0, sx, -1),
        (1, sx, 1),
        (2, sy, -1),
        (3, sy, 1),
    ):
        recurrent(out, out, dx=dx)
        p["Uh"][out] = source

    # Signed errors along the selected row.
    ex_f = -sx.copy()
    ex_f[11] -= 1.0
    ex_f[13] -= x_offset
    ex_h = np.zeros(16, dtype=np.float32)
    ex_h[0] = ex_h[1] = 1.0

    ey_f = -sy.copy()
    ey_f[12] -= 1.0
    ey_f[13] -= y_offset
    ey_h = np.zeros(16, dtype=np.float32)
    ey_h[2] = ey_h[3] = 1.0

    p["Uh"][4] = ex_f
    p["Wh"][4, :, 1, 1] = ex_h
    p["Uh"][5] = ey_f
    p["Wh"][5, :, 1, 1] = ey_h

    # Vertical broadcasts of coordinate totals.
    for out, source, left, right, dy in (
        (6, sx, 0, 1, -1),
        (7, sx, 0, 1, 1),
        (8, sy, 2, 3, -1),
        (9, sy, 2, 3, 1),
    ):
        recurrent(out, out, dy=dy)
        recurrent(out, left)
        recurrent(out, right)
        p["Uh"][out] = -source

    # Current anchor's Manhattan error and its broadcast.
    erow_f = -ex_f - ey_f
    erow_h = -ex_h - ey_h
    erow_h[4] += 2.0
    erow_h[5] += 2.0

    for out, dy in ((10, -1), (11, 1)):
        p["Uh"][out] = erow_f
        p["Wh"][out, :, 1, 1] = erow_h
        recurrent(out, out, dy=dy)

    error_f = -erow_f
    error_h = -erow_h
    error_h[10] += 1.0
    error_h[11] += 1.0

    # Candidate displacement from the proposed destination.
    dx_f = -sx.copy()
    dx_f[0] += 1.0
    dx_h = np.zeros(16, dtype=np.float32)
    dx_h[0] = dx_h[1] = 1.0
    dx_h[6] = dx_h[7] = -1.0

    dy_f = -sy.copy()
    dy_f[1] += 1.0
    dy_h = np.zeros(16, dtype=np.float32)
    dy_h[2] = dy_h[3] = 1.0
    dy_h[8] = dy_h[9] = -1.0

    p["Uh"][12] = dx_f
    p["Wh"][12, :, 1, 1] = dx_h
    p["bh"][12] = x_offset

    p["Uh"][13] = dy_f
    p["Wh"][13, :, 1, 1] = dy_h
    p["bh"][13] = y_offset

    base_f = dx_f + dy_f
    base_h = dx_h + dy_h
    base_h[12] -= 2.0
    base_h[13] -= 2.0
    base_b = x_offset + y_offset

    # Missing-goal flag.
    missing_f = dx_f.copy()
    missing_f[0] -= 1.0
    p["Uh"][15] = missing_f
    p["Wh"][15, :, 1, 1] = dx_h
    p["bh"][15] = x_offset - 4.0

    ban_cost = 16.0
    missing_cost = 4.0
    anchor_gate = 128.0

    # Inactive-anchor selection competes with improvement by movement.
    switch_f = -base_f - error_f
    switch_h = -base_h - error_h
    switch_f[17] += ban_cost
    switch_f[14] += anchor_gate
    switch_h[15] += missing_cost

    p["Uh"][14] = switch_f
    p["Wh"][14, :, 1, 1] = switch_h
    p["bh"][14] = -base_b + 1.0 / 63.0 - anchor_gate

    p["Wc"][0, :32] = base_f
    p["Wc"][0, 32:] = base_h
    p["bc"][0] = base_b

    p["Wc"][0, 17] -= ban_cost
    p["Wc"][0, 18] -= 128.0
    p["Wc"][0, 32 + 15] -= missing_cost
    p["Wc"][0, 32 + 14] += 1.0

    p["Wc"][0, 0] -= 0.001
    p["Wc"][0, 1] -= 0.01

    # No-motion watchdog. g[32:64] contains f3 spatial maxima.
    p["Wz"][0, 32 + 11] = 1.0
    p["Wz"][1, 32 + 12] = 1.0
    p["bz"][0:2] = 0.25

    # z[2] + z[3] is max(selected-anchor presence,
    # inactive-anchor presence), hence a recognition gate.
    p["Wz"][2, 32 + 13] = 1.0
    p["Wz"][3, 32 + 14] = 1.0
    p["Wz"][3, 32 + 13] = -1.0

    scale = 0.05
    tolerance = scale * 0.2 / 63.0
    sharpness = 30000.0

    p["Wmz"][0, 0] = scale
    p["Wmz"][1, 1] = scale

    for axis in range(2):
        positive = 2 + 2 * axis
        negative = positive + 1

        p["Wmz"][positive, axis] = sharpness * scale
        p["Wm"][positive, axis] = -sharpness
        p["bm"][positive] = -sharpness * tolerance

        p["Wmz"][negative, axis] = -sharpness * scale
        p["Wm"][negative, axis] = sharpness
        p["bm"][negative] = -sharpness * tolerance

    # Disable the watchdog immediately after RESET.
    p["Wma"][6, 0] = 4.0

    p["bo"][:] = -10.0
    p["bo"][6] = 0.0
    p["bo"][0] = -11.0

    p["Wo"][0, 2] = 8.0
    p["Wo"][0, 3] = 8.0
    for comparator in range(2, 6):
        p["Wo"][0, 128 + comparator] = -1.0
    p["Wo"][0, 128 + 6] = -16.0

    return p
