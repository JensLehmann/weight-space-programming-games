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
    adjacent = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def pixel(out, color, dy=0, dx=0, weight=1.0):
        W1[out, color, 2 + dy, 2 + dx] += weight

    # Goal centres, including covered goals.
    pixel(0, 11, weight=4)
    for dy, dx in adjacent:
        for color in (11, 5, 4):
            pixel(0, color, dy, dx)
    b1[0] = -7

    # Selected original-piece centres.
    for color in (0, 11):
        pixel(1, color, weight=4)
    for dy, dx in adjacent:
        pixel(1, 5, dy, dx)
    b1[1] = -7

    # Reflected-piece centres.
    for color in (4, 11):
        pixel(2, color, weight=4)
    for dy, dx in adjacent:
        pixel(2, 4, dy, dx)
    b1[2] = -7

    # Vertical and horizontal axis evidence.
    for d in range(-2, 3):
        pixel(3, 10, d, 0)
        pixel(4, 10, 0, d)
    b1[3] = -4
    b1[4] = -4

    vertical_pair = [
        (0, -1, 0), (0, 2, 0),
        (10, -1, -1), (10, -1, 1),
    ]
    horizontal_pair = [
        (0, 0, -1), (0, 0, 2),
        (10, -1, -1), (10, 1, -1),
    ]

    def pair_detector(out, tests, weight=1.0):
        for color, dy, dx in tests:
            pixel(out, color, dy, dx, weight)

    pair_detector(5, vertical_pair)
    pair_detector(6, horizontal_pair)
    b1[5] = -3
    b1[6] = -3

    # Selected-piece coordinate extrema.
    for out in (7, 8, 9, 10):
        for color in (0, 11):
            pixel(out, color, weight=4)
        for dy, dx in adjacent:
            pixel(out, 5, dy, dx)

    W1[7, 17, 2, 2] = -1
    b1[7] = -7
    W1[8, 17, 2, 2] = 1
    b1[8] = -8
    W1[9, 16, 2, 2] = -1
    b1[9] = -7
    W1[10, 16, 2, 2] = 1
    b1[10] = -8

    # Sparse markers near the beginning of selected axes.
    for out, tests, coord in [
        (15, horizontal_pair, 17),
        (17, vertical_pair, 16),
    ]:
        pair_detector(out, tests, 10)
        pair_detector(out + 1, tests, 10)
        W1[out, coord, 2, 2] = -100
        W1[out + 1, coord, 2, 2] = -100
        b1[out] = -35
        b1[out + 1] = -36

    # Selected-axis coordinate extrema.
    for out in (19, 20):
        pair_detector(out, vertical_pair)
    W1[19, 17, 2, 2] = -1
    b1[19] = -3
    W1[20, 17, 2, 2] = 1
    b1[20] = -4

    for out in (21, 22):
        pair_detector(out, horizontal_pair)
    W1[21, 16, 2, 2] = -1
    b1[21] = -3
    W1[22, 16, 2, 2] = 1
    b1[22] = -4

    W2 = p["W2"]
    for i in [0, 1, 2, 3, 4, 7, 8, 9, 10, 19, 20, 21, 22]:
        W2[i, i, 1, 1] = 1

    # Fill the gaps between successive selected-axis markers.
    for d in (-1, 0, 1):
        W2[5, 5, 1 + d, 1] = 1
        W2[6, 6, 1, 1 + d] = 1

    # Clip sparse marker values to [0, 1].
    W2[15, 15, 1, 1] = 1
    W2[15, 16, 1, 1] = -1
    W2[16, 17, 1, 1] = 1
    W2[16, 18, 1, 1] = -1

    W3, b3 = p["W3"], p["b3"]
    W3[0, 0] = 1
    W3[1, 1] = 1

    vertical_axis = 32 + 3
    horizontal_axis = 64 + 4
    selected_vertical = 32 + 5
    selected_horizontal = 64 + 6

    # Reflections whose horizontal motion is reversed.
    W3[2, 2] = 1
    W3[2, vertical_axis] = 1
    W3[2, selected_vertical] = -1
    W3[2, selected_horizontal] = -1
    b3[2] = -1

    # Reflections with unchanged horizontal motion.
    W3[3, 2] = 1
    W3[3, vertical_axis] = -1
    W3[3, selected_vertical] = -1
    W3[3, selected_horizontal] = -1

    W3[4, 2] = 1
    W3[4, selected_vertical] = 1
    b3[4] = -1

    # Corresponding masks for vertical motion.
    W3[5, 2] = 1
    W3[5, horizontal_axis] = 1
    W3[5, selected_vertical] = -1
    W3[5, selected_horizontal] = -1
    b3[5] = -1

    W3[6, 2] = 1
    W3[6, horizontal_axis] = -1
    W3[6, selected_vertical] = -1
    W3[6, selected_horizontal] = -1

    W3[7, 2] = 1
    W3[7, selected_horizontal] = 1
    b3[7] = -1

    # Sparse axis cues, suppressed on reflected pixels.
    W3[8, 16] = 1
    W3[8, 2] = -1
    W3[9, 15] = 1
    W3[9, 2] = -1

    for dst, src in zip(
        range(10, 18), (7, 8, 9, 10, 19, 20, 21, 22)
    ):
        W3[dst, src] = 1

    # Global selected-axis indicators via later spatial maxima.
    W3[18, 5] = 1
    W3[19, 6] = 1

    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    # Target counts per column.
    Wh[0, 0, 0, 1] = 1
    Wh[1, 1, 2, 1] = 1
    Uh[0, 0] = 1
    Uh[1, 0] = 1

    # Discounted left- and right-looking fields.
    Wh[2, 2, 1, 0] = 0.9
    Wh[3, 3, 1, 2] = 0.9
    for channel in (2, 3):
        Wh[channel, 0, 1, 1] = 1
        Wh[channel, 1, 1, 1] = 1
        Uh[channel, 0] = -1

    # Target counts per row.
    Wh[4, 4, 1, 0] = 1
    Wh[5, 5, 1, 2] = 1
    Uh[4, 0] = 1
    Uh[5, 0] = 1

    # Discounted up- and down-looking fields.
    Wh[6, 6, 0, 1] = 0.9
    Wh[7, 7, 2, 1] = 0.9
    for channel in (6, 7):
        Wh[channel, 4, 1, 1] = 1
        Wh[channel, 5, 1, 1] = 1
        Uh[channel, 0] = -1

    def gated_difference(channel, positive, negative, masks):
        Wh[channel, positive, 1, 1] = 1
        Wh[channel, negative, 1, 1] = -1
        for mask in masks:
            Uh[channel, mask] = 2048
        bh[channel] = -2048

    gated_difference(8, 2, 3, [1, 3, 4, 8])
    gated_difference(9, 3, 2, [1, 3, 4, 8])
    gated_difference(10, 2, 3, [2])
    gated_difference(11, 3, 2, [2])

    gated_difference(12, 6, 7, [1, 6, 7, 9])
    gated_difference(13, 7, 6, [1, 6, 7, 9])
    gated_difference(14, 6, 7, [5])
    gated_difference(15, 7, 6, [5])

    Wz, bz = p["Wz"], p["bz"]

    def add_sum(z, channel, weight):
        Wz[z, 16 + channel] += 4096 * weight

    # Directional scores: up, down, left, right.
    for c, w in [(12, 1), (13, -1), (15, 1), (14, -1)]:
        add_sum(0, c, w)
    for c, w in [(13, 1), (12, -1), (14, 1), (15, -1)]:
        add_sum(1, c, w)
    for c, w in [(8, 1), (9, -1), (11, 1), (10, -1)]:
        add_sum(2, c, w)
    for c, w in [(9, 1), (8, -1), (10, 1), (11, -1)]:
        add_sum(3, c, w)

    # Boundary indicators.
    for z, channel, threshold in [
        (4, 10, 0.970), (5, 11, 0.950),
        (6, 12, 0.970), (7, 13, 0.950),
        (8, 14, 0.970), (9, 15, 0.950),
        (10, 16, 0.970), (11, 17, 0.950),
    ]:
        Wz[z, 32 + channel] = 1
        bz[z] = -threshold

    Wz[12, 32 + 18] = 1  # selected vertical axis
    Wz[13, 32 + 19] = 1  # selected horizontal axis

    # Memory units 0:4 encode the previous action.
    for i in range(4):
        p["Wma"][i, 1 + i] = 4

    # Memory units 4:8 detect a reversal of the preceding action.
    for i, opposite in enumerate((1, 0, 3, 2)):
        p["Wm"][4 + i, i] = 12
        p["Wma"][4 + i, 1 + opposite] = 12
        p["bm"][4 + i] = -18

    Wo, bo = p["Wo"], p["bo"]
    bo[:] = -1_000_000

    for i in range(4):
        bo[1 + i] = 0
        Wo[1 + i, i] = 1

    for action, indicators in [
        (3, (4, 8)),
        (4, (5, 9)),
        (1, (6, 10)),
        (2, (7, 11)),
    ]:
        for indicator in indicators:
            Wo[action, indicator] = -5000

    # A vertical axis only moves left/right; a horizontal axis
    # only moves up/down.
    Wo[1, 12] = -100000
    Wo[2, 12] = -100000
    Wo[3, 13] = -100000
    Wo[4, 13] = -100000

    # Cycle selection at a weak local optimum or after a reversal.
    bo[5] = 3999.95
    for i in range(4):
        Wo[5, 128 + 4 + i] = 1000

    return p
