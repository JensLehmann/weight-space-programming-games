import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,),
        "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,),
        "Wh": (16, 16, 3, 3), "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,),
        "Wm": (64, 64), "Wmz": (64, 128),
        "Wma": (64, 8), "bm": (64,),
        "Wo": (8, 192), "bo": (8,),
        "Wc": (1, 48), "bc": (1,),
    }
    p = {k: np.zeros(s, np.float32) for k, s in shapes.items()}

    def c1(o, i, dy=0, dx=0, v=1.0):
        p["W1"][o, i, dy + 2, dx + 2] += v

    def c2(o, i, dy=0, dx=0, v=1.0):
        p["W2"][o, i, dy + 1, dx + 1] += v

    def rec(o, i, dy=0, dx=0, v=1.0):
        p["Wh"][o, i, dy + 1, dx + 1] += v

    def feat(o, terms, bias=0.0):
        for i, v in terms:
            p["W3"][o, i] += v
        p["b3"][o] = bias

    def scaled(terms, factor):
        return [(i, v * factor) for i, v in terms]

    # Wall cores: a black/grey region is blocked only when no
    # open-space, object, player, or exit pixel is nearby.
    p["b1"][0] = 1.0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            for color in (7, 8, 9, 10, 11, 12, 14, 15):
                c1(0, color, dy, dx, -1.0)

    # Supporting material, clipped in the following layer.
    for pos, neg, direction in ((2, 3, -1), (4, 5, 1)):
        for o, threshold in ((pos, 7.0), (neg, 8.0)):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy = direction + dy
                    for color in (3, 5, 12):
                        c1(o, color, yy, dx)
                    for color in (8, 14, 15):
                        c1(o, color, yy, dx, 9.0)
            p["b1"][o] = -threshold

    # Red/pink and green objects.
    c1(6, 14)
    c1(6, 15)
    c1(7, 8)

    # Exit, excluding the bottom action bar.
    c1(8, 7)
    c1(8, 16, v=-1.0)

    # Spike interiors and their black end markings.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            c1(9, 15, dy, dx)
            c1(9, 3, dy, dx, -64.0)
            c1(9, 0, dy, dx, 64.0)

    # Player body edges.
    for color in (9, 11):
        c1(10, color, 0, 2)
        c1(10, color, 0, 1, -1.0)
        c1(11, color, 0, -2)
        c1(11, color, 0, -1, -1.0)

    def direction_pair(pos, neg, source_y, upward):
        sign = 1.0 if upward else -1.0
        for o in (pos, neg):
            c1(o, 9, source_y, 0, 64.0)
            c1(o, 16, source_y, 0, 64.0 * sign)
        base = -96.0 if upward else -32.0
        p["b1"][pos] = base
        p["b1"][neg] = base - 1.0

    direction_pair(12, 13, 2, True)
    direction_pair(14, 15, -2, False)

    # Unique narrow body pixels distinguish the two camera/gravity
    # configurations, independently of which way the player faces.
    def canonical_pair(pos, neg, upward):
        adjacent = -1 if upward else 1
        pattern = (
            (9, 0, 0),
            (5, adjacent, 0),
            (9, -adjacent, 0),
            (5, 0, -1),
            (5, 0, 1),
        )
        sign = 1.0 if upward else -1.0
        base = -672.0 if upward else -608.0
        for o in (pos, neg):
            for color, dy, dx in pattern:
                c1(o, color, dy, dx, 128.0)
            c1(o, 16, v=64.0 * sign)
        p["b1"][pos] = base
        p["b1"][neg] = base - 1.0

    canonical_pair(16, 17, True)
    canonical_pair(18, 19, False)

    # Bottom exclusion ramp.
    c1(21, 16, v=63.0)
    p["b1"][21] = -58.0

    # Pink objects, excluding the action bar.
    for o, bias in ((22, -2.0), (23, -3.0)):
        c1(o, 15, v=64.0)
        c1(o, 16, v=-64.0)
        p["b1"][o] = bias

    # Player row coordinate, masked by its body colour.
    c1(24, 9, v=64.0)
    c1(24, 16)
    p["b1"][24] = -64.0

    # Progress rewards exist only on actual open/object pixels.
    for o, row_weight, bias in ((25, -1.0, -63.0),
                                (28, 1.0, -64.0)):
        for color in (7, 10, 12, 14, 15):
            c1(o, color, v=64.0)
        c1(o, 16, v=row_weight)
        p["b1"][o] = bias

    # Solid purple, excluding the sparse ghost template.
    for o, threshold in ((26, 5.0), (27, 6.0)):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                c1(o, 12, dy, dx)
        p["b1"][o] = -threshold

    # Player-row neighbourhood and the plain row coordinate.
    for dy in range(-2, 3):
        c1(29, 9, dy, 0)
    c1(30, 16)

    c1(31, 14)
    c1(31, 15)

    # Layer two.
    c2(0, 0)
    c2(1, 2, -1, 0)
    c2(1, 3, -1, 0, -1.0)
    c2(2, 4, 1, 0)
    c2(2, 5, 1, 0, -1.0)

    c2(3, 6)
    c2(3, 26)
    c2(3, 27, v=-1.0)
    c2(4, 7)
    c2(5, 8)

    for o in (6, 7):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                c2(o, 9, dy, dx)
    p["b2"][7] = -1.0

    for pos, neg, source, dx in ((8, 9, 10, 1),
                                 (10, 11, 11, -1)):
        c2(pos, source, 0, dx)
        c2(neg, source, 0, dx)
        p["b2"][neg] = -1.0

    for o, pos, neg, dy in (
        (12, 12, 13, 1), (13, 14, 15, -1),
        (14, 16, 17, 0), (15, 18, 19, 0),
    ):
        c2(o, pos, dy, 0)
        c2(o, neg, dy, 0, -1.0)

    c2(17, 21)
    c2(18, 22)
    c2(18, 23, v=-1.0)
    c2(19, 24)

    # Two ReLUs encode a clipped player-neighbourhood indicator.
    c2(20, 29)
    p["b2"][20] = -1.0
    c2(25, 29)

    for o in (21, 22):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                c2(o, 7, dy, dx)
    p["b2"][22] = -1.0

    c2(23, 26)
    c2(23, 27, v=-1.0)
    c2(24, 30)

    for o in (26, 27):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                c2(o, 31, dy, dx)
                c2(o, 26, dy, dx)
                c2(o, 27, dy, dx, -1.0)
    p["b2"][27] = -1.0

    for o in (28, 29):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                c2(o, 6, dy, dx)
                c2(o, 26, dy, dx)
                c2(o, 27, dy, dx, -1.0)
    p["b2"][29] = -1.0

    c2(30, 25)
    c2(31, 28)

    risk = [(6, 1.0), (7, -1.0)]
    green = [(21, 1.0), (22, -1.0)]
    left_edge = [(8, 1.0), (9, -1.0)]
    right_edge = [(10, 1.0), (11, -1.0)]
    solid = [(26, 1.0), (27, -1.0)]
    mutable = [(28, 1.0), (29, -1.0)]

    # Axial features: row maxima start at 32, column maxima at 64.
    feat(0, [(0, 1.0)])
    feat(1, [(1, -1.0)], 1.0)
    feat(2, [(2, -1.0)], 1.0)
    feat(3, green)
    feat(4, risk + [(17, -100.0)])
    feat(5, [(3, 1.0)])
    feat(6, [(5, 1.0)])

    # Penalize click candidates behind the player, but only in
    # the column containing the relevant canonical player marker.
    feat(7, [(24, 1.0), (78, 2.0), (83, -1.0)], -2.0)
    feat(8, [(83, 1.0), (24, -1.0), (79, 2.0)], -2.0)

    neighbourhood = [(57, 1.0), (52, -1.0)]
    feat(9, left_edge + neighbourhood, -1.0)
    feat(10, right_edge + neighbourhood, -1.0)
    feat(11, [(14, 1.0)])
    feat(12, [(15, 1.0)])

    # Click alignment can be satisfied by the player's column,
    # its row, or the immediate support projection. Paired ReLUs
    # clip the resulting mask to one.
    up_click = (
        mutable + [(78, 1.0), (46, 1.0), (12, 1.0)]
        + scaled(risk, -2.0) + [(17, -100.0)]
    )
    down_click = (
        mutable + [(79, 1.0), (47, 1.0), (13, 1.0)]
        + scaled(risk, -2.0) + [(17, -100.0)]
    )
    feat(13, up_click, -1.0)
    feat(17, up_click, -2.0)
    feat(14, down_click, -1.0)
    feat(29, down_click, -2.0)

    feat(15, [(18, 1.0)] + scaled(risk, -2.0)
         + [(17, -100.0)])
    feat(20, [(4, 1.0)])
    feat(21, [(23, 1.0)])
    feat(23, solid)

    for o, edge, direction in (
        (22, left_edge, 14), (24, right_edge, 14),
        (25, left_edge, 15), (26, right_edge, 15),
    ):
        feat(
            o,
            edge + [(32 + direction, 1.0), (0, -1.0)]
            + scaled(green, -1.0)
            + scaled(risk, -1.0)
            + scaled(solid, -1.0),
            -1.0,
        )

    # No reward for merely staying near the current player row.
    reward_exclusions = (
        [(57, -1.0), (52, 1.0), (17, -100.0)]
        + scaled(risk, -2.0)
    )
    feat(27, [(30, 1.0)] + reward_exclusions)
    feat(28, [(31, 1.0)] + reward_exclusions)

    barrier = 10000.0

    # Directional route fields. The sum of propagation weights is
    # below one, keeping values bounded without enormous gates.
    for horizontal, vertical, dy, unsupported, progress in (
        (0, 1, -1, 1, 27),
        (2, 3, 1, 2, 28),
    ):
        for source in (horizontal, vertical):
            rec(horizontal, source, 0, -1, 0.10)
            rec(horizontal, source, 0, 1, 0.10)
            rec(vertical, source, dy, 0, 0.78)

        for o in (horizontal, vertical):
            p["Uh"][o, 0] = -barrier
            p["Uh"][o, 3] = -barrier
            p["Uh"][o, 4] = -barrier
            p["Uh"][o, 6] = 10.0

        p["Uh"][horizontal, unsupported] = -barrier
        p["Uh"][horizontal, 23] = -barrier
        p["Uh"][vertical, progress] = 0.10

    # Visible straight-line spike danger, stopped by wall cores.
    for o, dy in ((14, -1), (15, 1)):
        rec(o, o, dy, 0, 0.99)
        p["Uh"][o, 4] = 1.0
        p["Uh"][o, 0] = -barrier

    def select(o, sources, mask, dx=0):
        for source, weight in sources:
            rec(o, source, 0, dx, weight)
        p["Uh"][o, mask] = barrier
        p["bh"][o] = -barrier

    # Removals/growth aligned with the player and current gravity.
    select(6, [(0, 1.0), (1, 1.0), (14, -100.0)], 13)
    p["Uh"][6, 17] = -barrier
    p["Uh"][6, 7] = -barrier

    select(7, [(2, 1.0), (3, 1.0), (15, -100.0)], 14)
    p["Uh"][7, 29] = -barrier
    p["Uh"][7, 8] = -barrier

    # Adjacent movement scores, sampling beyond the body edge.
    select(9, [(0, 1.0), (1, 1.0), (14, -100.0)], 22, -1)
    select(10, [(0, 1.0), (1, 1.0), (14, -100.0)], 24, 1)
    select(11, [(2, 1.0), (3, 1.0), (15, -100.0)], 25, -1)
    select(12, [(2, 1.0), (3, 1.0), (15, -100.0)], 26, 1)

    # Reversal requests require a better opposite route and no
    # visible spike on the opposite fall ray.
    select(
        4,
        [(2, 1.0), (3, 1.0), (0, -3.0), (1, -3.0),
         (15, -100.0)],
        11,
    )
    select(
        5,
        [(0, 1.0), (1, 1.0), (2, -3.0), (3, -3.0),
         (14, -100.0)],
        12,
    )

    # Carry reversal requests to the green click targets.
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rec(8, 8, dy, dx, 0.24)
    rec(8, 4)
    rec(8, 5)
    select(13, [(8, 1.0)], 20)

    def readout(o, terms):
        for i, v in terms:
            p["Wz"][o, i] += v

    readout(0, [(9, 1.0), (11, 1.0)])
    readout(1, [(10, 1.0), (12, 1.0)])
    readout(2, [(6, 1.0), (7, 1.0)])
    readout(3, [(13, 1000.0)])
    readout(4, [(47, 1.0)])
    readout(5, [(52, 1.0)])
    readout(6, [(54, 1.0), (57, 1.0)])
    readout(7, [(56, 1.0), (58, 1.0)])

    p["Wo"][3, 0] = 0.65
    p["Wo"][4, 1] = 0.65
    p["Wo"][6, 2] = 0.80
    p["Wo"][6, 3] = 0.80

    # Small fallbacks, below ordinary route values.
    p["Wo"][6, 4] = 1e-29
    p["Wo"][6, 5] = 1e-28
    p["Wo"][3, 6] = 1e-36
    p["Wo"][4, 7] = 1e-36
    p["bo"][4] = 1e-38

    # Persistent direction is only a tie-breaker.
    p["Wm"][0, 0] = 0.9
    p["Wma"][0, 3] = -1.0
    p["Wma"][0, 4] = 1.0
    p["Wo"][3, 128] = -1e-37
    p["Wo"][4, 128] = 1e-37

    for action in (0, 1, 2, 5, 7):
        p["bo"][action] = -1000.0

    # The click map uses the same candidate priorities as the
    # action readout, rather than an unrelated large multiplier.
    p["Wc"][0, 32 + 6] = 1.0
    p["Wc"][0, 32 + 7] = 1.0
    p["Wc"][0, 32 + 13] = 1000.0
    p["Wc"][0, 20] = 1e-28
    p["Wc"][0, 15] = 1e-29

    return p
