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

    colours = (7, 8, 9, 11, 12, 13, 14, 15)
    handle_colours = (4,) + colours

    diamond = [
        (dy, dx)
        for dy in range(-2, 3)
        for dx in range(-2, 3)
        if abs(dy) + abs(dx) <= 2
    ]
    arms = [q for q in diamond if q != (0, 0)]
    ball_shell = [
        (dy, dx)
        for dy in range(-2, 3)
        for dx in range(-2, 3)
        if not (abs(dy) == 2 and abs(dx) == 2)
        and (dy, dx) != (0, 0)
    ]
    square5 = [
        (dy, dx)
        for dy in range(-2, 3)
        for dx in range(-2, 3)
    ]

    def points(out, palette, offsets, weight=1.0):
        for dy, dx in offsets:
            for colour in palette:
                W1[out, colour, dy + 2, dx + 2] += weight

    # Exact diamond-centre matches. Correct templates yield one.
    for out, arm_colour in ((0, 0), (1, 3)):
        points(out, (arm_colour,), arms)
        points(out, handle_colours, ((0, 0),), 13.0)
        b1[out] = -24.0

    # Rounded ordinary ball with a colour-6 centre.
    points(2, colours, ball_shell)
    points(2, (6,), ((0, 0),), 21.0)
    b1[2] = -40.0

    # Four partial templates combine into a radius-three ring detector.
    partials = (
        ((-2, -1), (-2, 1), (-1, -2), (-1, 2)),
        ((2, -1), (2, 1), (1, -2), (1, 2)),
        ((-1, -2), (1, -2), (-2, -1), (2, -1)),
        ((-1, 2), (1, 2), (-2, 1), (2, 1)),
    )
    for j, offsets in enumerate(partials):
        points(3 + j, colours, offsets)
        b1[3 + j] = -3.0

    W1[7, 16, 2, 2] = 1.0       # row / 63
    W1[8, 17, 2, 2] = 1.0       # column / 63

    points(9, (2,), diamond)     # handle endpoint collision
    points(10, (0, 3), square5)  # possible handle-selection click

    # Keep complete handles visible and avoid incomplete boundary scans.
    W1[11, 16, 2, 2] = -63.0
    b1[11] = 2.0
    W1[12, 16, 2, 2] = 63.0
    b1[12] = -61.0
    W1[13, 17, 2, 2] = -63.0
    b1[13] = 2.0
    W1[14, 17, 2, 2] = 63.0
    b1[14] = -61.0

    points(15, (10,), ball_shell + [(0, 0)])

    def copy2(out, inp, weight=1.0):
        W2[out, inp, 1, 1] += weight

    def coordinate_gate(out, detector, coordinate):
        # At a detected centre this equals the coordinate; elsewhere zero.
        copy2(out, detector, 2.0)
        copy2(out, coordinate)
        b2[out] = -2.0

    copy2(0, 0)                 # selected handle
    copy2(1, 1)                 # inactive handle
    copy2(2, 0)
    copy2(2, 1)                 # either handle

    coordinate_gate(3, 0, 8)    # selected x
    coordinate_gate(4, 0, 7)    # selected y
    coordinate_gate(5, 2, 8)    # ball x
    coordinate_gate(6, 2, 7)    # ball y

    ring_reads = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def ring_sum(out, weight):
        for j, (dy, dx) in enumerate(ring_reads):
            W2[out, 3 + j, dy + 1, dx + 1] += weight

    ring_sum(7, 1.0)
    b2[7] = -3.0

    ring_sum(8, 2.0)
    copy2(8, 8)
    b2[8] = -8.0                # goal x

    ring_sum(9, 2.0)
    copy2(9, 7)
    b2[9] = -8.0                # goal y

    # Coordinates gated to either selected or inactive handle centres.
    for out, coordinate in ((10, 8), (11, 7)):
        copy2(out, 0, 2.0)
        copy2(out, 1, 2.0)
        copy2(out, coordinate)
        b2[out] = -2.0

    copy2(12, 8)
    copy2(13, 7)

    for inp in (9, 11, 12, 13, 14):
        copy2(14, inp)

    # Exclude clicks inside handle bounding boxes, except exact centres.
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            W2[15, 10, dy + 1, dx + 1] = 1.0
    copy2(15, 0, -1024.0)
    copy2(15, 1, -1024.0)

    copy2(16, 15)

    def feature(out, inp, weight=1.0, axis=None):
        offset = (
            0 if axis is None
            else 32 if axis == "row"
            else 64
        )
        W3[out, offset + inp] += weight

    # Axial selected-handle coordinate sources.
    feature(0, 3, 63.0, "col")
    feature(1, 4, 63.0, "row")

    # Destination A + (G - B), shifted positive before recurrent summation.
    # For n handles this reduces centroid error by approximately 1/n.
    offset = 256.0
    gain = 1.0

    b3[2] = offset
    feature(2, 3, 63.0, "col")
    feature(2, 8, 63.0 * gain, "col")
    feature(2, 5, -63.0 * gain, "col")

    b3[3] = offset
    feature(3, 4, 63.0, "row")
    feature(3, 9, 63.0 * gain, "row")
    feature(3, 6, -63.0 * gain, "row")

    feature(4, 12, 63.0)        # local x
    feature(5, 13, 63.0)        # local y
    feature(6, 2)              # handle centre
    feature(7, 10, 63.0)        # handle-centre x
    feature(8, 11, 63.0)        # handle-centre y
    feature(9, 0)              # selected centre
    feature(10, 14)            # blocked endpoint
    feature(11, 15)            # selection exclusion
    feature(12, 16)            # weak terrain preference

    b3[13] = 1.0
    feature(13, 2, -1.0)       # non-marker cell

    # Eight independent scans run in parallel.
    scans = (
        (0, 0, 1, 0),
        (1, 0, 1, 2),
        (2, 1, 0, 1),
        (3, 1, 2, 1),
        (4, 2, 1, 0),
        (5, 2, 1, 2),
        (6, 3, 0, 1),
        (7, 3, 2, 1),
    )
    for out, source, ky, kx in scans:
        Uh[out, source] = 1.0
        Wh[out, out, ky, kx] = 1.0

    # Broadcast selected coordinates, gated to handle centres.
    for out, prefix, suffix, source in (
        (8, 0, 1, 0),
        (9, 2, 3, 1),
    ):
        Wh[out, prefix, 1, 1] = 1.0
        Wh[out, suffix, 1, 1] = 1.0
        Uh[out, source] = -1.0
        Uh[out, 6] = 128.0
        bh[out] = -128.0

    # Ordinary cells: Manhattan distance to the proposed destination.
    #
    # Handle centres: off-screen penalty for the prospective destination
    # C + (G-B). This compares switching handles with a constrained move.
    def residual(out, prefix, suffix, source, coordinate,
                 marker_coordinate, gated_selected, sign):
        Wh[out, prefix, 1, 1] = -sign
        Wh[out, suffix, 1, 1] = -sign
        Uh[out, source] = sign
        bh[out] = sign * 64.0 * offset

        Uh[out, coordinate] = sign
        Wh[out, gated_selected, 1, 1] = sign

        if sign > 0:
            Uh[out, 6] = 2.0
            Uh[out, marker_coordinate] = -2.0
        else:
            Uh[out, 6] = -61.0
            Uh[out, marker_coordinate] = 2.0

    residual(10, 4, 5, 2, 4, 7, 8, 1.0)
    residual(11, 4, 5, 2, 4, 7, 8, -1.0)
    residual(12, 6, 7, 3, 5, 8, 9, 1.0)
    residual(13, 6, 7, 3, 5, 8, 9, -1.0)

    wc = p["Wc"][0]
    for channel in (10, 11, 12, 13):
        wc[32 + channel] = -1.0

    wc[6] = -12.0             # changing-handle cost
    wc[9] = -80.0             # do not reselect the active handle
    wc[10] = -10000.0          # reject walls and clipped destinations
    wc[11] = -1000.0           # reject accidental selection clicks
    wc[12] = -0.25
    wc[13] = -2.0

    p["bo"][6] = 1.0
    return p
