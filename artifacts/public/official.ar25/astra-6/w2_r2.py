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

    def w1(out, channel, dy=0, dx=0, value=1.0):
        p["W1"][out, channel, dy + 2, dx + 2] += value

    def w2(out, channel, dy=0, dx=0, value=1.0):
        p["W2"][out, channel, dy + 1, dx + 1] += value

    cross = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Palette masks.
    w1(0, 11)

    # A goal marker surrounded by occupied pixels.
    w1(1, 11, value=4.0)
    for dy, dx in cross:
        for color in (4, 5):
            w1(1, color, dy, dx)
    p["b1"][1] = -7.0

    for color in (4, 5):
        w1(2, color)

    # Restore the centre hole of an occupied game cell.
    for color in (0, 9, 11):
        w1(3, color, value=4.0)
    for dy, dx in cross:
        for color in (4, 5):
            w1(3, color, dy, dx)
    p["b1"][3] = -7.0

    # Energy-column and bottom-padding masks.
    w1(4, 17, value=126.0)
    p["b1"][4] = -125.0
    w1(5, 16, value=126.0)
    p["b1"][5] = -125.0

    w1(6, 16)
    w1(7, 17)

    # Mirror-line candidates.
    for dy, dx in [(0, -1), (0, 1), (-2, 0), (2, 0)]:
        w1(8, 10, dy, dx)
    p["b1"][8] = -3.0

    for dy, dx in [(-1, 0), (1, 0), (0, -2), (0, 2)]:
        w1(9, 10, dy, dx)
    p["b1"][9] = -3.0

    # Selected-axis candidates.
    w1(10, 0, value=2.0)
    w1(10, 10, -2, 0)
    w1(10, 10, 2, 0)
    p["b1"][10] = -3.0

    w1(11, 0, value=2.0)
    w1(11, 10, 0, -2)
    w1(11, 10, 0, 2)
    p["b1"][11] = -3.0

    # Shifted samples allow layer two to inspect adjacent game cells.
    cardinal = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    for i, (dy, dx) in enumerate(cardinal):
        w1(12 + i, 11, dy, dx)
        for color in (4, 5):
            w1(16 + i, color, dy, dx)

    diagonal = [(-2, -2), (2, 2), (-2, 2), (2, -2)]
    for i, (dy, dx) in enumerate(diagonal):
        w1(20 + i, 11, dy, dx)
        for color in (4, 5):
            w1(24 + i, color, dy, dx)

    w1(28, 0, -2, 0)
    w1(29, 0, 0, -2)

    # An uncovered selected-shape centre.
    w1(30, 0, value=4.0)
    for dy, dx in cross:
        w1(30, 5, dy, dx)
    p["b1"][30] = -7.0
    w1(31, 18)

    # Reconstruct goal cells. Channels 0 and 1 implement a capped sum.
    for out in (0, 1):
        w2(out, 0)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                w2(out, 1, dy, dx)
        w2(out, 4, value=-2.0)
        p["b2"][out] = -float(out)

    w2(2, 2)
    w2(2, 3)
    w2(2, 4, value=-2.0)
    w2(2, 5, value=-2.0)

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def corner(out, occupied, upper):
        base = 16 if occupied else 12
        w2(out, 2 if occupied else 0)
        if occupied:
            w2(out, 3)
        # North/east or south/east, excluding the opposite arms.
        positive_vertical = 0 if upper else 1
        negative_vertical = 1 if upper else 0
        for index, coefficient in (
            (positive_vertical, 1.0),
            (3, 1.0),
            (negative_vertical, -1.0),
            (2, -1.0),
        ):
            dy, dx = offsets[index]
            w2(out, base + index, dy, dx, coefficient)
        w2(out, 4, value=-2.0)
        w2(out, 5, value=-2.0)
        p["b2"][out] = -2.0

    corner(3, False, True)
    corner(4, False, False)
    corner(5, True, True)
    corner(6, True, False)

    w2(7, 8)
    w2(8, 9)

    # Require a second selected marker along an axis.
    w2(9, 10)
    w2(9, 28, -1, 0)
    p["b2"][9] = -1.0

    w2(10, 11)
    w2(10, 29, 0, -1)
    p["b2"][10] = -1.0

    def point_feature(out, terms, bias=0.0):
        for channel, coefficient in terms:
            w2(out, channel, value=coefficient)
        p["b2"][out] = bias

    # Axis and goal coordinate extrema.
    point_feature(11, [(8, 1), (7, 1)], -1)
    point_feature(12, [(8, 1), (7, -1)])
    point_feature(13, [(9, 1), (6, 1)], -1)
    point_feature(14, [(9, 1), (6, -1)])

    point_feature(15, [(0, 1), (6, 1), (4, -2)], -1)
    point_feature(16, [(0, 1), (6, -1), (4, -2)])
    point_feature(17, [(0, 1), (7, 1), (4, -2)], -1)
    point_feature(18, [(0, 1), (7, -1), (4, -2)])

    def square(out, occupied):
        w2(out, 2 if occupied else 0)
        if occupied:
            w2(out, 3)
        w2(out, 16 if occupied else 12, -1, 0)
        w2(out, 19 if occupied else 15, 0, 1)
        w2(out, 26 if occupied else 22, -1, 1)
        w2(out, 4, value=-2.0)
        w2(out, 5, value=-2.0)
        p["b2"][out] = -3.0

    square(19, False)
    square(21, True)

    w2(23, 30)
    point_feature(24, [(30, 1), (6, 1)], -1)
    point_feature(25, [(30, 1), (6, -1)])
    w2(26, 31)

    # Vertical runs distinguish long-arm corners from short thick pieces.
    for out, occupied in ((29, False), (30, True)):
        w2(out, 2 if occupied else 0)
        if occupied:
            w2(out, 3)
        w2(out, 16 if occupied else 12, -1, 0)
        w2(out, 17 if occupied else 13, 1, 0)
        w2(out, 4, value=-2.0)
        w2(out, 5, value=-2.0)
        p["b2"][out] = -2.0

    # Third-layer masks and geometric features.
    p["W3"][0, 0] = 1.0
    p["W3"][0, 1] = -1.0
    p["W3"][1, 2] = 1.0

    # Column maxima connect a corner to a vertical run in its column.
    for out, source, run in ((2, 3, 29), (3, 4, 29),
                             (4, 5, 30), (5, 6, 30)):
        p["W3"][out, source] = 1.0
        p["W3"][out, 64 + run] = 1.0
        p["b3"][out] = -1.0

    copies = {
        6: 7, 7: 8, 8: 9, 9: 10,
        10: 11, 11: 12, 12: 13, 13: 14,
        14: 15, 15: 16, 16: 17, 17: 18,
        18: 19, 20: 21,
        21: 23, 22: 24, 23: 25, 24: 26,
    }
    for out, source in copies.items():
        p["W3"][out, source] = 1.0

    # Four distance fields, using four recurrent channels each.
    # A-B is the reached mask; C accumulates arrival time.
    # D reads the remaining distance only at target-feature pixels.
    fields = [
        (1, 0, 126.0),
        (4, 2, 125.0),
        (5, 3, 125.0),
        (20, 18, 125.0),
    ]
    for group, (seed, target, radius) in enumerate(fields):
        a, b, c, d = [4 * group + i for i in range(4)]
        for out in (a, b):
            for dy, dx in [(0, 0)] + cross:
                p["Wh"][out, a, dy + 1, dx + 1] += 1.0
                p["Wh"][out, b, dy + 1, dx + 1] -= 1.0
            p["Uh"][out, seed] = 1.0
        p["bh"][b] = -1.0

        p["Wh"][c, c, 1, 1] = 1.0
        p["Wh"][c, a, 1, 1] = 1.0
        p["Wh"][c, b, 1, 1] = -1.0

        p["Wh"][d, c, 1, 1] = -1.0
        p["Uh"][d, target] = radius

    # g = [h_max, h_mean, f3_max, f3_mean].
    for unit, channel in enumerate((3, 7, 11, 15)):
        p["Wz"][unit, 16 + channel] = 1.0
        p["Wz"][unit, channel] = 0.04

    def fm(channel):
        return 32 + channel

    p["Wz"][10, fm(8)] = 1.0
    p["Wz"][11, fm(9)] = 1.0
    p["Wz"][16, fm(7)] = 1.0
    p["Wz"][17, fm(21)] = 1.0

    def axis_error(unit, goal_plus, goal_minus,
                   axis_plus, axis_minus, selected, sign):
        for channel, coefficient in (
            (goal_plus, sign),
            (goal_minus, -sign),
            (axis_plus, -sign),
            (axis_minus, sign),
        ):
            p["Wz"][unit, fm(channel)] += 31.5 * coefficient
        p["Wz"][unit, fm(selected)] += 100.0
        p["bz"][unit] = -101.6

    axis_error(12, 16, 17, 10, 11, 8, 1.0)
    axis_error(13, 16, 17, 10, 11, 8, -1.0)
    axis_error(14, 14, 15, 12, 13, 9, 1.0)
    axis_error(15, 14, 15, 12, 13, 9, -1.0)

    # Signed selected-piece centre minus horizontal-axis centre.
    centre_terms = [(22, 0.5), (23, -0.5),
                    (12, -0.5), (13, 0.5)]
    for channel, coefficient in centre_terms:
        p["Wz"][20, fm(channel)] = coefficient
        p["Wz"][21, fm(channel)] = -coefficient

    # Cost storage and comparison.
    cost_scale = 1.0e-8
    comparator_gain = 1.0e13
    for unit, weight in enumerate((1.0, 12.0, 12.0, 8.0)):
        p["Wmz"][0, unit] = cost_scale * weight
        p["Wmz"][1, unit] = comparator_gain * cost_scale * weight
    p["Wm"][1, 0] = -comparator_gain
    p["bm"][1] = comparator_gain * cost_scale / (4.0 * 4096.0)

    # Last-action indicators.
    for action in range(1, 5):
        p["Wma"][2 + action - 1, action] = 4.0
        p["Wma"][8, action] = 4.0

    # Last movement survives an undo.
    for j, action in enumerate(range(1, 5)):
        latch = 9 + j
        p["Wm"][latch, latch] = 5.0
        p["Wma"][latch, 1:5] = -12.0
        p["Wma"][latch, action] = 12.0

        after_undo = 13 + j
        p["Wm"][after_undo, latch] = 5.0
        p["Wma"][after_undo, 7] = 10.0
        p["bm"][after_undo] = -10.0

    # Escape-state memory:
    # 18: initialisation constant
    # 19: already escaped since the last selection
    # 20: side of horizontal mirror
    # 21,22: continuation tests with a crossing margin
    # 23,24: latched upward/downward escape modes
    p["bm"][18] = 10.0

    p["Wm"][19, 19] = 5.0
    p["Wm"][19, 18] = 5.0
    p["Wma"][19, 6] = 20.0
    p["Wma"][19, 5] = -40.0
    p["bm"][19] = -5.0

    for out, sign in ((20, 1.0), (21, 1.0), (22, -1.0)):
        p["Wmz"][out, 20] = 100.0 * sign
        p["Wmz"][out, 21] = -100.0 * sign
    p["bm"][21] = 100.0 * 5.0 / 63.0
    p["bm"][22] = 100.0 * 5.0 / 63.0

    for out, side_sign, continuation in (
        (23, 1.0, 21),
        (24, -1.0, 22),
    ):
        p["Wm"][out, out] = 20.0
        p["Wm"][out, 20] = 5.0 * side_sign
        p["Wm"][out, continuation] = 10.0
        p["Wma"][out, 6] = 20.0
        p["Wma"][out, 5] = -100.0
        p["bm"][out] = -10.0

    # Ordinary directional coordinate descent.
    p["bo"][0] = -10000.0
    p["bo"][1] = 1.0
    p["bo"][2] = 0.2
    p["bo"][3] = 0.1
    p["bo"][4] = 0.3

    for action in range(1, 5):
        p["Wo"][action, 128 + 2 + action - 1] = 10.0

    following = {1: 4, 4: 2, 2: 3, 3: 5}
    for previous, next_action in following.items():
        gate = 13 + previous - 1
        p["Wo"][next_action, 128 + gate] += 20.0
        p["bo"][next_action] += 20.0

    p["Wo"][7, 128 + 1] = 50.0
    p["Wo"][7, 128 + 8] = 100.0
    p["bo"][7] = -100.0

    # Align selected mirrors with the goal bounding-box centre.
    p["Wo"][5, 10] = 500.0
    p["Wo"][5, 11] = 500.0
    for action, error_unit in ((4, 12), (3, 13),
                               (2, 14), (1, 15)):
        p["Wo"][action, error_unit] = 1000.0

    # After a failed sweep, click the current piece to enter escape mode.
    p["Wo"][6, 128 + 16] = 100.0
    p["Wo"][6, 128 + 19] = -100.0
    p["Wo"][6, 16] = 100.0
    p["Wo"][6, 17] = 100.0
    p["Wo"][6, 10] = -1000.0
    p["Wo"][6, 11] = -1000.0
    p["bo"][6] = -350.0
    p["Wc"][0, 21] = 1.0

    # Escape movements override one-step cost rejection.
    p["Wo"][1, 128 + 23] = 1000.0
    p["Wo"][2, 128 + 24] = 1000.0
    p["bo"][1] += 1000.0
    p["bo"][2] += 1000.0

    return p
