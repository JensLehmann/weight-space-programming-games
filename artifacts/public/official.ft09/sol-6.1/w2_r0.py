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

    # Each canonical palette pair has one member in each group:
    # (9, 8), (8, 12), and (14, 15).
    groups = ((8, 14), (9, 12, 15))
    markers = (0, 2, 3)

    directions = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )
    cardinal = ((-1, 0), (1, 0), (0, -1), (0, 1))
    corners = ((-1, -1), (-1, 1), (1, -1), (1, 1))

    def input_weights(out, colors, dy=0, dx=0, weight=1.0):
        for color in colors:
            p["W1"][out, color, 2 + dy, 2 + dx] += weight

    def second_weight(out, inp, dy=0, dx=0, weight=1.0):
        p["W2"][out, inp, 1 + dy, 1 + dx] += weight

    # First-layer channels 0 and 1:
    # clue center of group A/B, with eight valid surrounding markers.
    for group_index, colors in enumerate(groups):
        input_weights(group_index, colors)
        for dy, dx in directions:
            input_weights(group_index, markers, 2 * dy, 2 * dx)
        p["b1"][group_index] = -8.0

    # Channels 2..9: equality markers in the eight directions.
    for d, (dy, dx) in enumerate(directions):
        input_weights(2 + d, (0,), 2 * dy, 2 * dx)

    # Channels 10 and 11: solid ordinary tiles of group A/B.
    for group_index, colors in enumerate(groups):
        ch = 10 + group_index
        input_weights(ch, colors)
        for dy, dx in directions:
            input_weights(ch, colors, 2 * dy, 2 * dx)
        p["b1"][ch] = -8.0

    # Channels 12 and 13: cross-toggle tiles of group A/B.
    for group_index, colors in enumerate(groups):
        ch = 12 + group_index
        input_weights(ch, colors)
        for dy, dx in corners:
            input_weights(ch, colors, 2 * dy, 2 * dx)
        for dy, dx in cardinal:
            input_weights(ch, (6,), 2 * dy, 2 * dx)
        p["b1"][ch] = -8.0

    # Channel 14: local tutorial-background evidence.
    for dy, dx in cardinal:
        input_weights(14, (5,), 2 * dy, 2 * dx)

    # Recognition initially occupies a 2x2 center patch. Subtracting
    # recognition immediately above and left leaves one spatial anchor.
    def subtract_preceding_clue_centers(out):
        for clue_channel in (0, 1):
            second_weight(out, clue_channel, -1, 0, -1.0)
            second_weight(out, clue_channel, 0, -1, -1.0)

    # Second-layer channels 0..15 encode the two ways to require A:
    # an A clue with an equality marker, or a B clue with inequality.
    for d in range(8):
        a_equal = 2 * d
        b_unequal = 2 * d + 1

        second_weight(a_equal, 0)
        second_weight(a_equal, 2 + d)
        subtract_preceding_clue_centers(a_equal)
        p["b2"][a_equal] = -1.0

        second_weight(b_unequal, 1)
        second_weight(b_unequal, 2 + d, weight=-1.0)
        subtract_preceding_clue_centers(b_unequal)

    # Channel 16: unique clue anchor, regardless of center group.
    second_weight(16, 0)
    second_weight(16, 1)
    subtract_preceding_clue_centers(16)

    # Channels 17..20: unique ordinary-A/B and cross-toggle-A/B anchors.
    for out, inp in zip(range(17, 21), range(10, 14)):
        second_weight(out, inp)
        second_weight(out, inp, -1, 0, -1.0)
        second_weight(out, inp, 0, -1, -1.0)

    # Channel 21 detects color 5 just outside a tile's center neighborhood.
    for dy, dx in cardinal:
        second_weight(21, 14, dy, dx)

    # Third-layer channels 0..15: directional requirements, interleaved A/B.
    # Requirement B is the complement of requirement A at a clue anchor.
    for d in range(8):
        require_a = 2 * d
        require_b = 2 * d + 1

        p["W3"][require_a, 2 * d] = 1.0
        p["W3"][require_a, 2 * d + 1] = 1.0

        p["W3"][require_b, 16] = 1.0
        p["W3"][require_b, 2 * d] = -1.0
        p["W3"][require_b, 2 * d + 1] = -1.0

    # Third-layer channels 16..19: interactive tile anchors.
    # Tutorial examples have color-5 surroundings and are suppressed.
    for out, inp in zip(range(16, 20), range(17, 21)):
        p["W3"][out, inp] = 1.0
        p["W3"][out, 21] = -1.0

    # Eight directions times two required colors.
    #
    # A unique source emits 766. Along an empty ray:
    #     h_next = relu(0.5 * h_previous - 1)
    # This gives amplitude 1 after eight pixels and zero after nine.
    # Damping also bounds reinforcement from collinear clue sources.
    p["bh"][:] = -1.0
    for d, (dy, dx) in enumerate(directions):
        for required_group in (0, 1):
            ch = 2 * d + required_group
            p["Wh"][ch, ch, 1 - dy, 1 - dx] = 0.5
            p["Uh"][ch, ch] = 767.0

            # Ordinary violations receive a small positive score;
            # cross-toggle violations retain approximately unit score.
            p["Uh"][ch, 16] -= 0.95
            p["Uh"][ch, 17] -= 0.95

            # Correct-colored endpoints extinguish the requirement.
            p["Uh"][ch, 16 + required_group] -= 2.0
            p["Uh"][ch, 18 + required_group] -= 2.0

    # Every interactive anchor outranks every non-anchor. Within anchors,
    # positive violation signals determine the selected click.
    for ch in range(16, 20):
        p["Wc"][0, ch] = 32768.0
    p["Wc"][0, 32:48] = 1.0

    # Always choose the click action when available.
    p["bo"][6] = 1.0

    return p
