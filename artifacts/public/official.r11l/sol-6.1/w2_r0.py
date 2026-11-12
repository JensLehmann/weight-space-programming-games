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

    W1, b1 = p["W1"], p["b1"]
    W2, b2 = p["W2"], p["b2"]
    W3, b3 = p["W3"], p["b3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    colours = (15, 14)
    corners = ((-2, -2), (-2, 2), (2, -2), (2, 2))
    cross = ((-1, 0), (1, 0), (0, -1), (0, 1))

    # W1 0..7: goal centres, ball centres, selected-handle
    # centres, inactive-handle centres; white/green pairs.
    for group, colour in enumerate(colours):
        for dy, dx in corners:
            W1[group, colour, dy + 2, dx + 2] = 2.0
        b1[group] = -7.0

        W1[2 + group, 6, 2, 2] = 2.0
        for dy, dx in cross:
            W1[2 + group, colour, dy + 2, dx + 2] = 2.0
        b1[2 + group] = -9.0

        for out, surrounding_colour in (
            (4 + group, 0),
            (6 + group, 3),
        ):
            W1[out, colour, 2, 2] = 2.0
            for dy, dx in cross:
                W1[out, surrounding_colour, dy + 2, dx + 2] = 2.0
            b1[out] = -9.0

    # Coordinate channels.
    W1[8, 16, 2, 2] = 1.0
    W1[9, 17, 2, 2] = 1.0

    # Exact handle-shaped endpoint collision test.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if abs(dx) + abs(dy) <= 2:
                W1[10, 2, dy + 2, dx + 2] = 1.0

    # First stage of the wider selected-handle proximity test.
    W1[11, 2, :, :] = 1.0

    # Selected-handle click-box exclusion.
    W1[12, 0, 1:4, 1:4] = 1.0

    # W2 0..7: coordinate-valued entity detections.
    # Entities are white goal, green goal, white ball, green ball.
    for entity in range(4):
        for axis, coordinate_channel in enumerate((8, 9)):
            out = 2 * entity + axis
            W2[out, entity, 1, 1] = 1.0
            W2[out, coordinate_channel, 1, 1] = 1.0
            b2[out] = -1.0

    # W2 8,9: selected-handle y,x.
    for out, coordinate_channel in ((8, 8), (9, 9)):
        W2[out, 4, 1, 1] = 1.0
        W2[out, 5, 1, 1] = 1.0
        W2[out, coordinate_channel, 1, 1] = 1.0
        b2[out] = -1.0

    # W2 10,11: selection indicators suppressed near walls.
    # A 3x3 neighbourhood of the 5x5 wall-count channel gives
    # a 7x7 effective footprint.
    for group in range(2):
        out = 10 + group
        W2[out, 4 + group, 1, 1] = 1.0
        W2[out, 11, :, :] = -1.0

    copies = {
        12: 0,
        13: 1,
        14: 4,
        15: 5,
        16: 6,
        17: 7,
        18: 10,
        19: 12,
        20: 8,
        21: 9,
    }
    for out, inp in copies.items():
        W2[out, inp, 1, 1] = 1.0

    def local(out, channel, coefficient=1.0):
        W3[out, channel] += coefficient

    def rowmax(out, channel, coefficient=1.0):
        W3[out, 32 + channel] += coefficient

    def colmax(out, channel, coefficient=1.0):
        W3[out, 64 + channel] += coefficient

    # F3 0..7: axial coordinate sources for goals and balls.
    for group in range(2):
        base = 4 * group
        goal_entity = group
        ball_entity = 2 + group

        colmax(base, 2 * goal_entity + 1)
        rowmax(base + 1, 2 * goal_entity)
        colmax(base + 2, 2 * ball_entity + 1)
        rowmax(base + 3, 2 * ball_entity)

    # F3 8,9: axial selected-handle x,y.
    colmax(8, 9)
    rowmax(9, 8)

    # Eligibility codes: goal plus wall-free selection in x,
    # goal plus actual selection in y.
    for group in range(2):
        colmax(10 + 2 * group, 12 + group)
        colmax(10 + 2 * group, 10 + group)
        rowmax(11 + 2 * group, 12 + group)
        rowmax(11 + 2 * group, 14 + group)

    rowmax(14, 14)
    rowmax(15, 15)

    local(16, 16)
    local(17, 17)
    local(18, 18)
    local(19, 19)
    local(20, 21)
    local(21, 20)

    # Endpoint centres must keep their sprites on screen.
    local(22, 21, 63.0)
    b3[22] = -60.0
    local(23, 21, -63.0)
    b3[23] = 3.0
    local(24, 20, 63.0)
    b3[24] = -60.0
    local(25, 20, -63.0)
    b3[25] = 3.0

    M = 32.0
    C = 72.0

    # Coordinate source encodes:
    # selected coordinate + 2*(goal coordinate - ball coordinate),
    # minus a large eligibility code.
    sources = []
    for group in range(2):
        base = 4 * group
        sx = np.zeros(32, dtype=np.float32)
        sy = np.zeros(32, dtype=np.float32)

        sx[base] = 2.0
        sx[base + 2] = -2.0
        sx[8] = 1.0
        sx[10 + 2 * group] = -M

        sy[base + 1] = 2.0
        sy[base + 3] = -2.0
        sy[9] = 1.0
        sy[11 + 2 * group] = -M

        sources.extend((sx, sy))

    # H 0..7: opposing prefix scans. Once converged, adding
    # opposing scans and subtracting the local source produces
    # the source's global sum plus the known constant 65*C.
    directions = ((0, -1), (0, 1), (-1, 0), (1, 0))
    for group in range(2):
        for j, (dy, dx) in enumerate(directions):
            out = 4 * group + j
            Wh[out, out, dy + 1, dx + 1] = 1.0
            Uh[out] = sources[2 * group + j // 2]
            bh[out] = C

    # H 8..11: positive coordinate-error ramps.
    for group in range(2):
        for axis in range(2):
            out = 8 + 2 * group + axis
            first = 4 * group + 2 * axis

            Wh[out, first, 1, 1] = -1.0
            Wh[out, first + 1, 1, 1] = -1.0
            Uh[out] = sources[2 * group + axis]
            Uh[out, 20 + axis] += 1.0
            bh[out] = 65.0 * C - 2.0 * M

    # H 12,13: broadcast selected colour, white +1, green -1.
    colour_source = np.zeros(32, dtype=np.float32)
    colour_source[14] = 1.0
    colour_source[15] = -1.0
    colour_offset = 2.0

    for out, dy in ((12, -1), (13, 1)):
        Wh[out, out, dy + 1, 1] = 1.0
        Uh[out] = colour_source
        bh[out] = colour_offset

    def add_decoded_sum(out, field, coefficient):
        first = 2 * field
        Wh[out, first, 1, 1] += coefficient
        Wh[out, first + 1, 1, 1] += coefficient
        Uh[out] -= coefficient * sources[field]
        bh[out] -= coefficient * 65.0 * C

    B = 4096.0

    # H 14,15: handle-selection bonuses.
    # The x-y code discrepancy signals wall-limited selection.
    # The other group's code and selected colour also permit
    # switching away from a completed goal.
    for group in range(2):
        out = 14 + group

        for field, sign in enumerate((1.0, -1.0, 1.0, -1.0)):
            add_decoded_sum(out, field, B * 5.0 * sign / M)

        other = 1 - group
        add_decoded_sum(out, 2 * other + 1, B / M)

        colour_sign = -1.5 if group == 0 else 1.5
        Wh[out, 12, 1, 1] += B * colour_sign
        Wh[out, 13, 1, 1] += B * colour_sign
        Uh[out] -= B * colour_sign * colour_source
        bh[out] -= B * colour_sign * 65.0 * colour_offset

        # Restrict the bonus to actual inactive-handle centres.
        Uh[out, 16 + group] += 8.0 * B
        bh[out] -= 8.25 * B

    # Sharp endpoint objective:
    # q - 2*relu(q-target), independently for x and y.
    p["Wc"][0, 20] = 63.0
    p["Wc"][0, 21] = 63.0
    for channel in (8, 9, 10, 11):
        p["Wc"][0, 32 + channel] = -126.0

    p["Wc"][0, 32 + 14] = 1.0
    p["Wc"][0, 32 + 15] = 1.0

    # Ordinary endpoint moves should not accidentally select handles.
    p["Wc"][0, 16] = -128.0
    p["Wc"][0, 17] = -128.0

    for channel in (18, 19, 22, 23, 24, 25):
        p["Wc"][0, channel] = -100000.0

    p["bo"][6] = 1.0
    return p
