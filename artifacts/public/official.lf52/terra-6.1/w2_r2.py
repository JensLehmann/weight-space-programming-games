import math
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

    def template(channel, positive, negative=()):
        for dx, dy, colour in positive:
            p["W1"][channel, colour, dy + 2, dx + 2] += 2.0
        for dx, dy, colour in negative:
            p["W1"][channel, colour, dy + 2, dx + 2] -= 2.0
        p["b1"][channel] = 1.0 - 2.0 * len(positive)

    circle = [
        (0, -1), (1, -1),
        (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
    ]
    outside = [
        (-1, -1), (2, -1), (-2, 0), (-1, 2), (2, 2)
    ]

    for channel, colour in [(0, 14), (1, 8), (2, 9)]:
        template(
            channel,
            [(x, y, colour) for x, y in circle],
            [(x, y, colour) for x, y in outside],
        )

    template(3, [
        (-1, 0, 15), (2, 0, 15),
        (0, -2, 15), (1, -2, 15),
        (0, 0, 7), (1, 0, 7), (1, -1, 7),
        (0, -1, 0),
        (0, 2, 5), (1, 2, 5),
    ])

    interior = [(x, y) for y in range(-1, 3) for x in range(-1, 3)]
    template(
        4,
        [(x, y, 1) for x, y in interior]
        + [(-2, 0, 0), (0, -2, 0)],
    )
    template(
        5,
        [(x, y, 12) for x, y in interior]
        + [(-2, 0, 11), (0, -2, 11)],
    )

    outline = [
        (0, -1), (1, -1), (-1, 0), (2, 0),
        (-1, 1), (2, 1), (0, 2), (1, 2),
    ]
    for channel, centre in [(6, 1), (7, 12)]:
        template(
            channel,
            [(x, y, 2) for x, y in outline] + [(0, 0, centre)],
        )

    for channel, colour in [(8, 14), (9, 8), (10, 9)]:
        template(
            channel,
            [(x, y, colour) for x, y in circle]
            + [(0, -2, 3), (-2, 0, 3)],
            [(x, y, colour) for x, y in outside],
        )

    template(11, [
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
        (-1, 0, 5), (2, 0, 5),
        (0, -1, 5), (1, -1, 5), (0, 2, 5), (1, 2, 5),
    ])

    p["W1"][12, 16, 2, 2] = 1.0
    p["W1"][13, 17, 2, 2] = 1.0

    # Rail-tile edge anchors, ordered up, down, left, right.
    # Each occupied template identifies a green peg on a yellow rail tile.
    occupied = [
        [(-2, -1, 11), (0, 1, 14), (1, 1, 14)],
        [(-2, 1, 11), (0, -2, 14), (1, -2, 14)],
        [(-1, -2, 11), (1, 0, 14), (1, 1, 14)],
        [(1, -2, 11), (-2, 0, 14), (-2, 1, 14)],
    ]
    platform = [
        [(-2, -1, 11), (-1, -1, 11), (-2, 0, 11)],
        [(-2, 1, 11), (-1, 1, 11), (-2, 0, 11)],
        [(-1, -2, 11), (0, -2, 11), (-1, -1, 11)],
        [(1, -2, 11), (0, -2, 11), (1, -1, 11)],
    ]
    tracks = [
        [(0, -2, 5), (1, -2, 5)],
        [(0, 2, 5), (1, 2, 5)],
        [(-2, 0, 5), (-2, 1, 5)],
        [(2, 0, 5), (2, 1, 5)],
    ]
    for i in range(4):
        template(14 + i, occupied[i])
        template(18 + i, tracks[i])
        template(22 + i, platform[i])

    for i in range(14):
        p["W2"][i, i, 1, 1] = 1.0

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i, (dy, dx) in enumerate(offsets):
        # Rail continuing beyond an occupied tile.
        p["W2"][14 + i, 14 + i, 1, 1] = 1.0
        p["W2"][14 + i, 18 + i, 1 + dy, 1 + dx] = 1.0
        p["b2"][14 + i] = -1.0

        # Rail continuing beyond any unselected tile.
        p["W2"][18 + i, 22 + i, 1, 1] = 1.0
        p["W2"][18 + i, 18 + i, 1 + dy, 1 + dx] = 1.0
        p["b2"][18 + i] = -1.0

    p["W2"][22, 12, 1, 1] = 1.0
    p["W2"][23, 13, 1, 1] = 1.0

    for i in (4, 5, 6, 7):
        p["W3"][0, i] = 1.0
    for i in (0, 1, 2):
        p["W3"][1, i] = 1.0
        p["W3"][2, i] = 1.0
    p["W3"][3, 6] = 1.0
    p["W3"][3, 7] = 1.0
    p["W3"][4, 11] = 1.0
    p["W3"][5, 0] = 1.0
    p["W3"][6, 1] = 1.0
    p["W3"][7, 2] = 1.0
    p["W3"][8, 8] = 1.0
    p["W3"][9, 9] = 1.0
    p["W3"][10, 10] = 1.0
    p["W3"][13, 3] = 1.0
    for i in (8, 9, 10):
        p["W3"][14, i] = 1.0

    # Coordinates of landing markers.
    for out, coord in [(15, 12), (16, 13)]:
        p["W3"][out, 6] = 2.0
        p["W3"][out, 7] = 2.0
        p["W3"][out, coord] = 1.0
        p["b3"][out] = -2.0

    for i in range(4):
        p["W3"][20 + i, 14 + i] = 1.0
        p["W3"][24 + i, 18 + i] = 1.0

    # Green-peg coordinates.
    for out, coord in [(28, 23), (29, 22)]:
        p["W3"][out, 0] = 2.0
        p["W3"][out, coord] = 1.0
        p["b3"][out] = -2.0

    # Coordinates of rail tiles with at least one detected continuation.
    for out, coord in [(30, 23), (31, 22)]:
        for i in range(18, 22):
            p["W3"][out, i] = 2.0
        p["W3"][out, coord] = 1.0
        p["b3"][out] = -2.0

    # Clock and pulse for six-pixel propagation.
    p["Wh"][0, 0, 1, 1] = 1.0
    p["bh"][0] = 1.0

    p["Uh"][1, 0] = 1.0
    p["Wh"][1, 0, 1, 1] = -2.0

    p["Wh"][2, 0, 1, 1] = -1.0
    p["bh"][2] = 7.0
    p["Wh"][3, 0, 1, 1] = 1.0
    p["bh"][3] = -6.0
    p["Wh"][4, 0, 1, 1] = -1.0
    p["bh"][4] = 14.0
    p["Wh"][5, 0, 1, 1] = 1.0
    p["bh"][5] = -13.0

    for i, (dy, dx) in enumerate(offsets):
        wave = 6 + i
        relay = 10 + i

        p["Wh"][wave, wave, 1 + dy, 1 + dx] = 1.0
        p["Wh"][wave, 1, 1, 1] = 1.0
        p["Wh"][wave, 3, 1, 1] = -2.0

        p["Wh"][relay, relay, 1 + dy, 1 + dx] = 2.0
        p["Wh"][relay, wave, 1, 1] = 1.0
        p["Wh"][relay, 2, 1, 1] = -100.0
        p["Wh"][relay, 5, 1, 1] = -1000.0
        p["Uh"][relay, 1] = 1.7
        p["Uh"][relay, 13] = 1.0 if (dy > 0 or dx > 0) else 0.35
        p["bh"][relay] = -1.0

    # Latch all jump sources and separately the selected jump source.
    for out, feature in [(14, 2), (15, 14)]:
        p["Wh"][out, out, 1, 1] = 1.0
        for relay in range(10, 14):
            p["Wh"][out, relay, 1, 1] = 1.0
        p["Wh"][out, 4, 1, 1] = -100.0
        p["Uh"][out, feature] = 100.0
        p["bh"][out] = -100.0

    p["Wz"][0, 14] = 1.0
    p["Wz"][1, 15] = 1.0
    p["Wz"][2, 32 + 14] = 1.0
    p["Wz"][3, 32 + 4] = 1.0

    for i in range(4):
        p["Wz"][4 + i, 32 + 20 + i] = 1.0
        p["Wz"][8 + i, 32 + 24 + i] = 1.0

    p["Wz"][12, 32 + 28] = 1.0
    p["Wz"][13, 32 + 29] = 1.0
    p["Wz"][14, 32 + 30] = 1.0
    p["Wz"][15, 32 + 31] = 1.0

    # Two small oscillators break exploration ties.
    for base, angle, bias in [
        (0, math.pi / 12.0, 0.025),
        (2, math.pi / 7.0, 0.018),
    ]:
        gain = 1.03
        c = gain * math.cos(angle)
        s = gain * math.sin(angle)
        p["Wm"][base, base] = c
        p["Wm"][base, base + 1] = -s
        p["Wm"][base + 1, base] = s
        p["Wm"][base + 1, base + 1] = c
        p["bm"][base] = bias

    # Store the preceding rail action, to discourage immediate reversal.
    for i, action in enumerate((1, 2, 3, 4)):
        p["Wma"][8 + i, action] = 2.0

    p["bo"][:] = -100.0
    for i, action in enumerate((1, 2, 3, 4)):
        p["bo"][action] = 0.0
        p["Wo"][action, 4 + i] = 4.0
        p["Wo"][action, 8 + i] = 3.0

    # Discourage reversing the previous control.
    for action, previous in [(1, 2), (2, 1), (3, 4), (4, 3)]:
        p["Wo"][action, 128 + 8 + (previous - 1)] = -0.8

    # Move unoccupied rail tiles toward the visible green region.
    for action, sign, green, tile in [
        (1, -1.0, 13, 15),
        (2, 1.0, 13, 15),
        (3, -1.0, 12, 14),
        (4, 1.0, 12, 14),
    ]:
        p["Wo"][action, green] += 2.0 * sign
        p["Wo"][action, tile] -= 2.0 * sign

    p["bo"][2] += 0.15
    p["bo"][4] += 0.20
    p["Wo"][1, 128 + 1] = 0.35
    p["Wo"][1, 128 + 3] = 0.35
    p["Wo"][2, 128 + 1] = -0.35
    p["Wo"][2, 128 + 3] = -0.35
    p["Wo"][3, 128 + 0] = 0.35
    p["Wo"][3, 128 + 2] = 0.35
    p["Wo"][4, 128 + 0] = -0.35
    p["Wo"][4, 128 + 2] = -0.35

    # A valid jump overrides navigation; a selected source needs an
    # admissible landing before the policy clicks its marker.
    p["bo"][6] = -1.0
    p["Wo"][6, 0] = 15.0
    p["Wo"][6, 1] = 20.0
    p["Wo"][6, 2] = -20.0
    p["Wo"][6, 3] = 30.0

    p["Wc"][0, 3] = 100.0
    p["Wc"][0, 4] = 1000.0
    p["Wc"][0, 5] = 0.15
    p["Wc"][0, 6] = 0.10
    p["Wc"][0, 15] = 1.0
    p["Wc"][0, 16] = 1.0
    p["Wc"][0, 32 + 14] = 1.0

    return p
