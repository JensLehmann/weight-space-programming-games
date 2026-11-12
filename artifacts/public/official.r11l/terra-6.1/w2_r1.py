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
    W3 = p["W3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    # Normalized row and column.
    W1[0, 16, 2, 2] = 1
    W1[1, 17, 2, 2] = 1

    # Colour-15 goal.
    for dy in (-2, 2):
        for dx in (-2, 2):
            W1[2, 15, 2 + dy, 2 + dx] = 1
    b1[2] = -3

    # Colour-15 ball.
    W1[3, 6, 2, 2] = 2
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        W1[3, 15, 2 + dy, 2 + dx] = 2
    b1[3] = -9

    diamond = [
        (dy, dx)
        for dy in range(-2, 3)
        for dx in range(-2, 3)
        if 0 < abs(dy) + abs(dx) <= 2
    ]

    # Selected 15, unselected 15, and selected 14 anchors.
    for out, border, centre in ((4, 0, 15), (5, 3, 15), (6, 0, 14)):
        for dy, dx in diamond:
            W1[out, border, 2 + dy, 2 + dx] = 2
        W1[out, centre, 2, 2] = 2
        b1[out] = -25

    # Visible wall pixels in a prospective anchor footprint.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if abs(dy) + abs(dx) <= 2:
                W1[7, 2, 2 + dy, 2 + dx] = 1

    # Distance outside the fully visible anchor-centre region.
    lo = 2.0 / 63.0
    hi = 61.0 / 63.0
    for out, coordinate in ((8, 16), (10, 17)):
        W1[out, coordinate, 2, 2] = -1
        b1[out] = lo
    for out, coordinate in ((9, 16), (11, 17)):
        W1[out, coordinate, 2, 2] = 1
        b1[out] = -hi

    W1[12, 0, 2, 2] = 1

    def center(out, inp, weight=1.0):
        W2[out, inp, 1, 1] = weight

    # Coordinate at a detected centre.
    for out, detector, coordinate in (
        (0, 2, 0),  # goal row
        (1, 2, 1),  # goal column
        (4, 4, 0),  # selected row
        (5, 4, 1),  # selected column
    ):
        center(out, detector)
        center(out, coordinate)
        b2[out] = -1

    for out, detector, coordinate in (
        (2, 3, 0),  # one minus ball row
        (3, 3, 1),  # one minus ball column
    ):
        center(out, detector)
        center(out, coordinate, -1)

    center(6, 0)
    center(7, 1)
    center(8, 7)
    center(9, 5)
    center(10, 6)

    # Selected-anchor neighbourhood.
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            W2[11, 4, 1 + dy, 1 + dx] = 1

    for out, inp in ((12, 8), (13, 9), (14, 10), (15, 11), (16, 12)):
        center(out, inp)

    ROW, COL = 32, 64

    # Target for a two-anchor ball:
    # selected + 2 * (goal - ball). The encoded source includes +2.
    W3[0, ROW + 0] = 2
    W3[0, ROW + 2] = 2
    W3[0, ROW + 4] = 1

    W3[1, COL + 1] = 2
    W3[1, COL + 3] = 2
    W3[1, COL + 5] = 1

    W3[2, 6] = 1
    W3[3, 7] = 1

    W3[4, ROW + 4] = 1
    W3[4, ROW + 10] = 2
    W3[5, COL + 5] = 1

    W3[6, 9] = 1
    W3[7, 8] = 1
    W3[8, 11] = 1

    for inp in (12, 13, 14, 15):
        W3[9, inp] = 1
    W3[10, 16] = 1

    # Bidirectional sweeps broadcast target-coordinate sources.
    for out, dy, dx, source in (
        (0, 0, 1, 0),
        (1, 2, 1, 0),
        (2, 1, 0, 1),
        (3, 1, 2, 1),
    ):
        Wh[out, out, dy, dx] = 1
        Uh[out, source] = 1

    # Positive and negative parts of target-coordinate error.
    for out, sign in ((4, 1), (5, -1)):
        Wh[out, 0, 1, 1] = sign
        Wh[out, 1, 1, 1] = sign
        Uh[out, 0] = -sign
        Uh[out, 2] = -sign
        bh[out] = -2 * sign

    for out, sign in ((6, 1), (7, -1)):
        Wh[out, 2, 1, 1] = sign
        Wh[out, 3, 1, 1] = sign
        Uh[out, 1] = -sign
        Uh[out, 3] = -sign
        bh[out] = -2 * sign

    # Broadcast the selected-anchor coordinate.
    for out, dy, dx, source in (
        (12, 0, 1, 4),
        (13, 2, 1, 4),
        (14, 1, 0, 5),
        (15, 1, 2, 5),
    ):
        Wh[out, out, dy, dx] = 1
        Uh[out, source] = 1

    # Switching scores, gated to unselected colour-15 anchors.
    #
    # A lower-bound score combines target overflow, the prospective
    # anchor's relative position, and the selected anchor's edge
    # proximity. Upper-bound scores use the corresponding mirror image.
    gate = 12.0
    margin = 0.15
    relative = 0.5
    edge_weight = 8.0
    overflow_weight = 5.0

    def lower_score(out, target_pair, target_source,
                    selected_pair, selected_source, candidate_source):
        a = overflow_weight
        s = relative + edge_weight
        for c in target_pair:
            Wh[out, c, 1, 1] = -a
        Uh[out, target_source] = a
        for c in selected_pair:
            Wh[out, c, 1, 1] = -s
        Uh[out, selected_source] = s
        Uh[out, candidate_source] = relative
        Uh[out, 6] = gate
        bh[out] = 2 * a + edge_weight * margin - gate

    def upper_score(out, target_pair, target_source,
                    selected_pair, selected_source, candidate_source):
        a = overflow_weight
        s = relative + edge_weight
        for c in target_pair:
            Wh[out, c, 1, 1] = a
        Uh[out, target_source] = -a
        for c in selected_pair:
            Wh[out, c, 1, 1] = s
        Uh[out, selected_source] = -s
        Uh[out, candidate_source] = -relative
        Uh[out, 6] = gate
        bh[out] = (
            -2 * a
            - a
            - edge_weight * (1 - margin)
            - gate
        )

    lower_score(8, (0, 1), 0, (12, 13), 4, 2)
    upper_score(9, (0, 1), 0, (12, 13), 4, 2)
    lower_score(10, (2, 3), 1, (14, 15), 5, 3)
    upper_score(11, (2, 3), 1, (14, 15), 5, 3)

    p["bo"][6] = 1

    # Click-map score.
    for channel in (4, 5, 6, 7):
        p["Wc"][0, 32 + channel] = -1
    for channel in (8, 9, 10, 11):
        p["Wc"][0, 32 + channel] = 8

    p["Wc"][0, 7] = -3
    p["Wc"][0, 8] = -3
    p["Wc"][0, 9] = -100
    p["Wc"][0, 10] = -0.03

    return p
