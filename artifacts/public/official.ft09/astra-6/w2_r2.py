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

    # Binary encoding separating each canonical palette pair:
    # (9, 8), (8, 12), and (14, 15).
    groups = ((8, 14), (9, 12, 15))
    marks = (0, 2, 3)
    directions = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    def input_weight(out, colors, dy, dx, weight=1.0):
        for color in colors:
            p["W1"][out, color, dy + 2, dx + 2] += weight

    def clue_detector(out, group, zero_direction=None):
        # At scale two, these tests select the upper-left pixel
        # of the clue's central 2x2 colour block.
        input_weight(out, group, 0, 0)
        input_weight(out, marks, -1, 0)
        input_weight(out, marks, 0, -1)
        p["b1"][out] = -2.0

        if zero_direction is not None:
            dy, dx = zero_direction
            input_weight(out, (0,), 2 * dy, 2 * dx)
            p["b1"][out] = -3.0

    # f1[0:2]: clue anchors, separated by centre-colour class.
    # f1[2:18]: those anchors additionally requiring a zero mark
    # in one specified direction.
    for side, group in enumerate(groups):
        clue_detector(side, group)
        for d, direction in enumerate(directions):
            clue_detector(2 + 8 * side + d, group, direction)

    # f1[18:20]: playable-looking tile interiors, allowing the
    # colour-6 markings of multi-tile switches.
    # f1[23:25]: ordinary, unmarked tile interiors.
    for side, group in enumerate(groups):
        marked = 18 + side
        ordinary = 23 + side

        for dy in range(-2, 3):
            for dx in range(-2, 3):
                input_weight(marked, group + (6,), dy, dx)
                input_weight(ordinary, group, dy, dx)

        input_weight(marked, group, 0, 0)
        p["b1"][marked] = -25.0
        p["b1"][ordinary] = -24.0

    # Background samples used by layer two to identify exact
    # tile centers and reject the noninteractive tutorial examples.
    input_weight(20, (4,), -2, 0)
    input_weight(21, (4,), 0, -2)

    def second_weight(out, src, dy=0, dx=0, weight=1.0):
        p["W2"][out, src, dy + 1, dx + 1] += weight

    for channel in range(18):
        second_weight(channel, channel)

    def tile_anchor(out, interior):
        second_weight(out, interior)
        second_weight(out, 20, -1, 0)
        second_weight(out, 21, 0, -1)
        p["b2"][out] = -2.0

    tile_anchor(18, 18)  # All playable class-A tiles.
    tile_anchor(19, 19)  # All playable class-B tiles.
    tile_anchor(20, 23)  # Ordinary class-A tiles.
    tile_anchor(21, 24)  # Ordinary class-B tiles.

    # A blocker immediately upstream of every clue prevents
    # persistent sources from accumulating along a propagation ray.
    for d, (dy, dx) in enumerate(directions):
        second_weight(22 + d, 0, dy, dx)
        second_weight(22 + d, 1, dy, dx)

    # f3[0:16]: directional requirements.
    # Zero means "same as clue"; every other mark means "different".
    for d in range(8):
        a_zero = 2 + d
        b_zero = 10 + d

        # Require A: (A and zero) or (B and nonzero).
        p["W3"][d, a_zero] = 1.0
        p["W3"][d, 1] = 1.0
        p["W3"][d, b_zero] = -1.0

        # Require B: (B and zero) or (A and nonzero).
        p["W3"][8 + d, b_zero] = 1.0
        p["W3"][8 + d, 0] = 1.0
        p["W3"][8 + d, a_zero] = -1.0

        p["W3"][16 + d, 22 + d] = 1.0

    p["W3"][24, 18] = 1.0
    p["W3"][25, 19] = 1.0

    # Ordinary-tile mask.
    p["W3"][26, 20] = 1.0
    p["W3"][26, 21] = 1.0

    # All clickable tile anchors.
    p["W3"][27, 18] = 1.0
    p["W3"][27, 19] = 1.0

    # Each recurrent channel is a directed, linearly decaying ray.
    # A source has height 9. Eight pixels away—the adjacent tile
    # center—it has height 1. Beyond that it vanishes.
    for side in range(2):
        for d, (dy, dx) in enumerate(directions):
            channel = 8 * side + d
            p["Wh"][channel, channel, 1 - dy, 1 - dx] = 1.0

            p["bh"][channel] = -1.0
            p["Uh"][channel, channel] = 10.0
            p["Uh"][channel, 16 + d] = -100.0

            # Suppress requirements already satisfied at a tile.
            p["Uh"][channel, 24 + side] = -100.0

            # Ordinary mismatches score 0.1 per constraint;
            # marked-switch mismatches score 1.0. Thus any
            # constrained wrong switch precedes ordinary repairs.
            p["Uh"][channel, 26] = -0.9

    # Restrict the click argmax to playable anchors, then rank
    # them by their unsatisfied incoming requirements.
    p["Wc"][0, 27] = 1000.0
    p["Wc"][0, 32:48] = 1.0

    # Always use the game's click action.
    p["bo"][:] = -1.0
    p["bo"][6] = 1.0

    return p
