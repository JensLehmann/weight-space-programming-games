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

    # ---------------------------------------------------------------
    # First convolution: visual templates and elementary features.
    # ---------------------------------------------------------------

    player = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 0, 0, 1],
    ], dtype=np.int8)

    target = np.array([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1],
    ], dtype=np.int8)

    def rotate_point(x, y, size, turns):
        for _ in range(turns):
            x, y = size - 1 - y, x
        return x, y

    def colour_template(unit, bitmap, marker):
        mx, my = marker
        height, width = bitmap.shape
        positives = 0

        for dy in range(-2, 3):
            for dx in range(-2, 3):
                xx = mx + dx
                yy = my + dy
                occupied = (
                    0 <= xx < width
                    and 0 <= yy < height
                    and bitmap[yy, xx] != 0
                )
                p["W1"][unit, 11, dy + 2, dx + 2] = (
                    2.0 if occupied else -2.0
                )
                positives += int(occupied)

        p["b1"][unit] = 1.0 - 2.0 * positives

    # Channels 0..3 recognize the ordinary player.
    player_markers = []
    for rotation in range(4):
        bitmap = np.rot90(player, -rotation)
        marker = rotate_point(2, 2, 4, rotation)
        colour_template(rotation, bitmap, marker)
        player_markers.append((rotation, marker, False))

    # Channels 4,28,29,30 recognize the doubled player's notch.
    doubled = np.repeat(np.repeat(player, 2, axis=0), 2, axis=1)
    doubled_units = [4, 28, 29, 30]

    for rotation, unit in enumerate(doubled_units):
        bitmap = np.rot90(doubled, -rotation)
        marker = rotate_point(3, 6, 8, rotation)
        colour_template(unit, bitmap, marker)
        player_markers.append((unit, marker, True))

    # Channels 5..8 recognize the target.
    target_markers = []
    for rotation in range(4):
        bitmap = np.rot90(target, -rotation)
        marker = rotate_point(2, 3, 6, rotation)
        unit = 5 + rotation
        colour_template(unit, bitmap, marker)
        target_markers.append((unit, marker))

    p["W1"][9, 16, 2, 2] = 1.0
    p["W1"][10, 17, 2, 2] = 1.0

    def bar_template(unit, vertical=False, first_horizontal=False):
        if vertical:
            filled = [(0, -1), (0, 0), (0, 1)]
            surround = [
                (-1, -1), (-1, 0), (-1, 1),
                (1, -1), (1, 0), (1, 1),
                (0, -2), (0, 2),
            ]
        else:
            filled = [(-1, 0), (0, 0), (1, 0)]
            surround = [
                (-1, -1), (0, -1), (1, -1),
                (-1, 1), (0, 1), (1, 1),
                (-2, 0), (2, 0),
            ]

        for dx, dy in filled:
            p["W1"][unit, 1, dy + 2, dx + 2] += 2.0
            p["W1"][unit, 5, dy + 2, dx + 2] += 2.0

        for dx, dy in surround:
            p["W1"][unit, 0, dy + 2, dx + 2] += 2.0

        p["b1"][unit] = (
            1.0 - 2.0 * (len(filled) + len(surround))
        )

        if first_horizontal:
            p["W1"][unit, 0, 0, 3] -= 2.0

    # 11: horizontal stroke; 12: first horizontal stroke;
    # 13: vertical stroke; 14: current on-state.
    bar_template(11)
    bar_template(12, first_horizontal=True)
    bar_template(13, vertical=True)
    p["W1"][14, 5, 2, 2] = 1.0

    # Solid Run-button interior, excluding outlined selector buttons.
    p["W1"][15, 9, 1:4, 1:4] = 1.0
    p["b1"][15] = -8.0

    # Row hinges select the relevant editor strokes.
    hinge_rows = [
        35, 36, 37,
        44, 45, 46,
        47, 48, 49,
        41, 42, 43,
    ]
    hinge_unit = {}

    for unit, row in enumerate(hinge_rows, start=16):
        hinge_unit[row] = unit
        p["W1"][unit, 16, 2, 2] = 63.0
        p["b1"][unit] = -float(row)

    p["W1"][31, 18, 2, 2] = 1.0

    # ---------------------------------------------------------------
    # Second convolution: coordinates, bit states, and row identities.
    # ---------------------------------------------------------------

    def second(unit, terms, bias=0.0):
        for source, weight in terms.items():
            p["W2"][unit, source, 1, 1] += float(weight)
        p["b2"][unit] = float(bias)

    def player_coordinate(unit, axis):
        terms = {10 if axis == 0 else 9: 1.0}

        for source, marker, is_doubled in player_markers:
            offset = marker[axis]
            # On leftward routes, this reserves a leading slot that
            # the scale-bit family turns into a shrink instruction.
            extra = 4 if axis == 0 and is_doubled else 0
            terms[source] = 10.0 + (extra - offset) / 63.0

        second(unit, terms, -10.0)

    def target_coordinate(unit, axis):
        terms = {10 if axis == 0 else 9: 1.0}
        for source, marker in target_markers:
            terms[source] = 10.0 - (marker[axis] - 1) / 63.0
        second(unit, terms, -10.0)

    player_coordinate(0, 0)
    target_coordinate(1, 0)
    player_coordinate(2, 1)
    target_coordinate(3, 1)

    # Left and right image edges.
    p["W2"][4, 31, 1, 0] = -1.0
    p["b2"][4] = 1.0
    p["W2"][5, 31, 1, 2] = -1.0
    p["b2"][5] = 1.0

    second(6, {11: 1.0})

    def add_peak(terms, centre):
        for row, coefficient in [
            (centre - 1, 1.0),
            (centre, -2.0),
            (centre + 1, 1.0),
        ]:
            source = hinge_unit[row]
            terms[source] = terms.get(source, 0.0) + coefficient

    def masked_bit_pair(off_unit, on_unit, stroke, centres=None):
        for unit, on in [(off_unit, False), (on_unit, True)]:
            terms = {
                stroke: 1.0,
                14: 1.0 if on else -1.0,
            }

            if centres is None:
                bias = -1.0 if on else 0.0
            else:
                for centre in centres:
                    add_peak(terms, centre)
                bias = -2.0 if on else -1.0

            second(unit, terms, bias)

    masked_bit_pair(7, 8, 12)
    masked_bit_pair(9, 10, 13, [36, 45])
    masked_bit_pair(11, 12, 13, [48])
    masked_bit_pair(15, 16, 13, [42])

    # Any currently-on editor stroke.
    second(17, {11: 1.0, 13: 1.0, 14: 1.0}, -1.0)
    second(18, {15: 1.0})
    second(19, {12: 1.0})
    second(20, {unit: 1.0 for unit in doubled_units})

    def stroke_identity(unit, stroke, centres):
        terms = {stroke: 1.0}
        for centre in centres:
            add_peak(terms, centre)
        second(unit, terms, -1.0)

    # These identities remain unchanged when bits are toggled.
    stroke_identity(21, 13, [36, 45])
    stroke_identity(22, 13, [48])
    stroke_identity(23, 13, [42])

    # ---------------------------------------------------------------
    # Axial layer: multiplex coordinate packets by instruction-bit row.
    # ---------------------------------------------------------------

    packet_gate = 8.0

    def projected_packet(unit, feature, identity):
        p["W3"][unit, 64 + feature] = 1.0
        p["W3"][unit, 32 + identity] = packet_gate
        p["b3"][unit] = -packet_gate

    for offset, identity in [(0, 19), (4, 21), (8, 22)]:
        for feature in range(4):
            projected_packet(offset + feature, feature, identity)

    projected_packet(12, 20, 23)

    p["W3"][13, 4] = 1.0
    p["W3"][14, 5] = 1.0
    p["W3"][15, 64 + 6] = 1.0

    p["W3"][16, 7] = 1.0
    p["W3"][17, 8] = 1.0
    p["W3"][18, 9] = 1.0
    p["W3"][19, 10] = 1.0
    p["W3"][20, 11] = 1.0
    p["W3"][20, 15] = 1.0
    p["W3"][21, 12] = 1.0
    p["W3"][21, 16] = 1.0

    # On bits outside the constructed movement/scale instruction set.
    p["W3"][22, 17] = 1.0
    for source in [8, 10, 12, 16]:
        p["W3"][22, source] = -1.0

    p["W3"][23, 18] = 1.0

    # Effective variables for the interval families:
    # bit 0: x=dy, y=-dx
    # bit 1: x=dy, y=dx
    # bit 5: x=dx, y=dy
    # bit 3: x=0,  y=-doubled_flag
    x_sources = {
        3: 1.0, 2: -1.0,
        7: 1.0, 6: -1.0,
        9: 1.0, 8: -1.0,
    }
    y_sources = {
        0: 1.0, 1: -1.0,
        5: 1.0, 4: -1.0,
        11: 1.0, 10: -1.0,
        12: -4.0 / 63.0,
    }

    # ---------------------------------------------------------------
    # Tied block: signed broadcasts, program rank, interval mismatches.
    # ---------------------------------------------------------------

    def recurrent(destination, source, weight, dx=0, dy=0):
        p["Wh"][destination, source, dy + 1, dx + 1] += weight

    def drive(destination, source, weight):
        p["Uh"][destination, source] += weight

    # Opposing prefix sums broadcast signed variables without losing
    # negative contributions to the ReLU.
    for left_unit, right_unit, sources in [
        (0, 1, x_sources),
        (2, 3, y_sources),
    ]:
        recurrent(left_unit, left_unit, 1.0, dx=-1)
        recurrent(right_unit, right_unit, 1.0, dx=1)
        drive(left_unit, 13, 2.0)
        drive(right_unit, 14, 2.0)

        for source, weight in sources.items():
            drive(left_unit, source, weight)
            drive(right_unit, source, weight)

    # One-based program-column rank.
    recurrent(4, 4, 1.0, dx=-1)
    drive(4, 15, 1.0)

    coordinate_to_cells = 63.0 / 4.0

    def add_variable(destination, coefficient, axis):
        if axis == "x":
            first, second_unit = 0, 1
            sources = x_sources
        else:
            first, second_unit = 2, 3
            sources = y_sources

        factor = coefficient * coordinate_to_cells
        recurrent(destination, first, factor)
        recurrent(destination, second_unit, factor)
        p["bh"][destination] -= 4.0 * factor

        # Opposing prefixes double-count the local source.
        for source, weight in sources.items():
            drive(destination, source, -factor * weight)

    add_variable(5, -1.0, "x")
    add_variable(6, -1.0, "y")

    # With L=relu(-x), U=relu(-y), k=j+1, an interval's
    # signed membership margin is:
    # upper-k+0.5 - relu(lower+upper-2*k+1).
    interval_specs = [
        (7, 1.0, 1.0, 2.0, 3.0),
        (8, 1.0, 1.0, 1.0, 3.0),
        (9, 2.0, 0.0, 4.0, 1.0),
    ]

    for unit, cx, cy, c_left, c_up in interval_specs:
        add_variable(unit, cx, "x")
        add_variable(unit, cy, "y")
        recurrent(unit, 5, c_left)
        recurrent(unit, 6, c_up)
        recurrent(unit, 4, -2.0)
        p["bh"][unit] += 1.0

    mismatch_gate = 128.0

    output_specs = [
        # off, on, hinge, x, y, L, U, off_mask, on_mask
        (10, 11, 7, 1.0, 1.0, 2.0, 2.0, 16, 17),
        (12, 13, 8, 1.0, 1.0, 1.0, 2.0, 18, 19),
        (14, 15, 9, 1.0, 0.0, 2.0, 1.0, 20, 21),
    ]

    for (
        off_unit, on_unit, hinge,
        cx, cy, c_left, c_up, off_mask, on_mask
    ) in output_specs:
        for unit, sign, mask in [
            (off_unit, 1.0, off_mask),
            (on_unit, -1.0, on_mask),
        ]:
            add_variable(unit, sign * cx, "x")
            add_variable(unit, sign * cy, "y")
            recurrent(unit, 5, sign * c_left)
            recurrent(unit, 6, sign * c_up)
            recurrent(unit, 4, -sign)
            recurrent(unit, hinge, -sign)
            p["bh"][unit] += sign * 0.5

            drive(unit, mask, mismatch_gate)
            p["bh"][unit] -= mismatch_gate

    # ---------------------------------------------------------------
    # Heads: edit mismatches first; otherwise click Run.
    # ---------------------------------------------------------------

    p["bo"][6] = 1.0
    p["bo"][0] = -1.0

    p["Wc"][0, 22] = 1.0
    p["Wc"][0, 23] = 0.1

    for channel in range(10, 16):
        p["Wc"][0, 32 + channel] = 1.0

    return p
