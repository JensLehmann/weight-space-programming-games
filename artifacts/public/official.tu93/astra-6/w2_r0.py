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

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    opposite = [1, 0, 3, 2]

    decay = float(np.float32(0.8))
    lane_subtraction = float(np.float32(0.999))
    gate = 1.0e6
    floor_signal = 1.0e-30

    def w1(out, channel, dy, dx, value):
        p["W1"][out, channel, dy + 2, dx + 2] += value

    def w2(out, channel, value=1.0, dy=0, dx=0):
        p["W2"][out, channel, dy + 1, dx + 1] += value

    def wh(out, channel, dy, dx, value):
        p["Wh"][out, channel, dy + 1, dx + 1] += value

    # First convolution:
    # 0 road skeleton; 1 player center; 2 exit; 3 node centers.
    # 4..7 red/patrol heading sources.
    # 8..11 active purple heading sources.
    # 12,13 coordinates; 14..17 patrol-only heading sources.
    road_colors = (0, 2, 4, 8, 9, 11, 12, 13, 14, 15)
    node_colors = tuple(c for c in road_colors if c != 2)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            for color in road_colors:
                w1(0, color, dy, dx, 1.0)
            for color in node_colors:
                w1(3, color, dy, dx, 1.0)
            w1(2, 14, dy, dx, 1.0)

    p["b1"][[0, 2, 3]] = -8.0

    for dy, dx in ((0, 0), (-1, -1), (-1, 1),
                   (1, -1), (1, 1)):
        w1(1, 9, dy, dx, 1.0)
    p["b1"][1] = -4.0

    for d, (dy, dx) in enumerate(directions):
        py, px = -dx, dy
        my, mx = -2 * dy, -2 * dx

        detectors = (
            (4 + d, (8, 12), (11, 15)),
            (8 + d, (13,), (11,)),
            (14 + d, (12,), (11, 15)),
        )
        for out, bodies, markers in detectors:
            for color in markers:
                w1(out, color, my, mx, 1.0)
            for sign in (-1, 1):
                for color in bodies:
                    w1(out, color, my + sign * py,
                       mx + sign * px, 1.0)
            for color in bodies:
                w1(out, color, -dy, -dx, -1.0)
            p["b1"][out] = -2.0

    w1(12, 16, 0, 0, 1.0)
    w1(13, 17, 0, 0, 1.0)

    # Second convolution:
    # 0..3 masks; 4..7 all relevant heading sources.
    # 8..11 player/exit coordinates.
    # 12,13 vertical/horizontal patrol indicators.
    # 14,15 absence of vertical/horizontal road continuation.
    # 16 non-node mask.
    for i in range(4):
        w2(i, i)

    for d in range(4):
        w2(4 + d, 4 + d)
        w2(4 + d, 8 + d)

    for out, center, coordinate in (
        (8, 1, 12), (9, 1, 13),
        (10, 2, 12), (11, 2, 13),
    ):
        w2(out, center)
        w2(out, coordinate)
        p["b2"][out] = -1.0

    w2(12, 14)
    w2(12, 15)
    w2(13, 16)
    w2(13, 17)

    w2(14, 0, -1.0, dy=-1)
    w2(14, 0, -1.0, dy=1)
    p["b2"][14] = 1.0

    w2(15, 0, -1.0, dx=-1)
    w2(15, 0, -1.0, dx=1)
    p["b2"][15] = 1.0

    w2(16, 3, -1.0)
    p["b2"][16] = 1.0

    # Third layer:
    # 0 road; 1 wall-or-player; 2 exit; 3..6 heading sources.
    # 7 non-player; 8,9 vertical/horizontal patrol-lane nodes.
    # 10..13 coordinates; 14 intersection of lane-node masks.
    p["W3"][0, 0] = 1.0
    p["W3"][1, 0] = -1.0
    p["W3"][1, 1] = 1.0
    p["b3"][1] = 1.0
    p["W3"][2, 2] = 1.0

    for d in range(4):
        p["W3"][3 + d, 4 + d] = 1.0

    p["W3"][7, 1] = -1.0
    p["b3"][7] = 1.0

    # Row maxima start at 32; column maxima start at 64.
    vertical_presence = 64 + 12
    horizontal_presence = 32 + 13

    p["W3"][8, vertical_presence] = 1.0
    p["W3"][8, 14] = -1.0
    p["W3"][8, 16] = -1.0

    p["W3"][9, horizontal_presence] = 1.0
    p["W3"][9, 15] = -1.0
    p["W3"][9, 16] = -1.0

    p["W3"][14, vertical_presence] = 1.0
    p["W3"][14, horizontal_presence] = 1.0
    p["W3"][14, 14] = -1.0
    p["W3"][14, 15] = -1.0
    p["W3"][14, 16] = -2.0
    p["b3"][14] = -1.0

    for i in range(4):
        p["W3"][10 + i, 8 + i] = 1.0

    # Recurrent channels:
    # 0..3 directional navigation messages.
    # 4..7 navigation copies gated to patrol-lane nodes.
    # 8..11 short, oriented hazard ramps.
    # 12..15 player-gated action values.
    for d, (dy, dx) in enumerate(directions):
        for incoming in range(4):
            if incoming != opposite[d]:
                wh(d, incoming, -dy, -dx, decay)
                wh(d, 4 + incoming, -dy, -dx,
                   -decay * lane_subtraction)

        # At a threatened location, reverse navigation may leave
        # only toward the enemy's current position.
        for heading in range(4):
            if d != opposite[heading]:
                wh(d, 8 + heading, -dy, -dx, -gate)

        p["Uh"][d, 0] = floor_signal
        p["Uh"][d, 1] = -gate
        p["Uh"][d, 2] = 1.0

        # Union of the two lane masks, without double counting.
        wh(4 + d, d, 0, 0, 1.0)
        p["Uh"][4 + d, 8] = gate
        p["Uh"][4 + d, 9] = gate
        p["Uh"][4 + d, 14] = -gate
        p["bh"][4 + d] = -gate

        # Heading detector is three pixels ahead of the enemy.
        # Values 4,3,2,1 extend through its next destination.
        wh(8 + d, 8 + d, -dy, -dx, 1.0)
        p["Uh"][8 + d, 3 + d] = 5.0
        p["bh"][8 + d] = -1.0

        # Read reverse flux from the neighbor in action direction d.
        for incoming in range(4):
            if incoming != d:
                wh(12 + d, incoming, dy, dx, decay)
                wh(12 + d, 4 + incoming, dy, dx,
                   -decay * lane_subtraction)

        for heading in range(4):
            if heading != d:
                wh(12 + d, 8 + heading, dy, dx, -gate)

        p["Uh"][12 + d, 7] = -gate

    # The tiny uniform drive distinguishes a five-pixel dead end
    # from a complete six-pixel move, even at a costly lane node.
    c = decay
    q5 = 3.0 * c * (1.0 - c**5) / (1.0 - c)
    q6 = 3.0 * c * (1.0 - c**6) / (1.0 - c)
    residual = 1.0 - lane_subtraction
    gap = residual * (q6 - q5)
    low = q5 + 0.1 * gap
    high = q5 + 0.3 * gap
    span = high - low

    unsafe_penalty = 1.0e4
    goal_preference = 1.0e-23
    reversal_penalty = 1.0e-24

    p["bo"][:] = -gate
    p["bo"][1:5] = 0.0

    for d in range(4):
        pooled = 12 + d
        action = 1 + d

        p["Wz"][d, pooled] = 1.0
        p["Wo"][action, d] = 1.0

        # Difference of downward hinges: bounded unsafe indicator.
        scale = -1.0 / (floor_signal * span)
        p["Wz"][4 + d, pooled] = scale
        p["bz"][4 + d] = high / span
        p["Wz"][8 + d, pooled] = scale
        p["bz"][8 + d] = low / span
        p["Wo"][action, 4 + d] = -unsafe_penalty
        p["Wo"][action, 8 + d] = unsafe_penalty

    # Pooled f3 maxima begin at g[32].
    player_row, player_col = 42, 43
    exit_row, exit_col = 44, 45
    coordinate_pairs = (
        (player_row, exit_row),
        (exit_row, player_row),
        (player_col, exit_col),
        (exit_col, player_col),
    )
    for d, (positive, negative) in enumerate(coordinate_pairs):
        p["Wz"][12 + d, positive] = 1.0
        p["Wz"][12 + d, negative] = -1.0
        p["Wo"][1 + d, 12 + d] = goal_preference
        p["Wo"][1 + d, 12 + opposite[d]] = -goal_preference

    # Weak anti-reversal tie-breaker, subordinate to navigation.
    for d in range(4):
        p["Wma"][d, 1 + d] = 4.0
        p["Wo"][1 + opposite[d], 128 + d] = -reversal_penalty

    return p
