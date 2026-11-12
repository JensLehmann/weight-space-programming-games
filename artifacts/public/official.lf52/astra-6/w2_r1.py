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
    p = {name: np.zeros(shape, np.float32)
         for name, shape in shapes.items()}

    pegs = (14, 8, 9)
    ordinary = (14, 8)
    jump_dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    arrow_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    actions = [1, 2, 3, 4]
    post_first = [6, 27, 28, 29]
    preferred = [23, 24, 25, 31]

    def f1(out, colors, y, x, weight=1.0):
        if isinstance(colors, int):
            colors = (colors,)
        for color in colors:
            p["W1"][out, color, y + 2, x + 2] += weight

    def f2(out, channel, y=0, x=0, weight=1.0):
        p["W2"][out, channel, y + 1, x + 1] += weight

    def rec(out, channel, y=0, x=0, weight=1.0):
        p["Wh"][out, channel, y + 1, x + 1] += weight

    corners = {(-1, -1), (-1, 2), (2, -1), (2, 2)}
    middle = {(0, 0), (0, 1), (1, 0), (1, 1)}
    disk = [
        (y, x)
        for y in range(-1, 3)
        for x in range(-1, 3)
        if (y, x) not in corners
    ]

    # Anchors use the upper-left pixel of a tile's central 2x2.
    def disk_detector(out, colors):
        for y, x in disk:
            f1(out, colors, y, x)
        for y, x in corners:
            f1(out, colors, y, x, -1)
        p["b1"][out] = -11

    disk_detector(0, ordinary)
    disk_detector(1, 9)
    disk_detector(24, 2)
    disk_detector(30, 14)

    # Purple posts.
    for y, x in ((0, -1), (0, 2), (-2, 0), (1, 0)):
        f1(2, 15, y, x)
    f1(2, 7, 0, 0)
    p["b1"][2] = -4

    # Empty white and orange tiles.
    for out, color in ((3, 1), (4, 12)):
        for y in range(-1, 3):
            for x in range(-1, 3):
                f1(out, color, y, x)
        p["b1"][out] = -15

    # Displayed landing rings, excluding filled gray disks.
    for y, x in disk:
        f1(5, 2, y, x, -1 if (y, x) in middle else 1)
    for y, x in corners:
        f1(5, 2, y, x, -1)
    p["b1"][5] = -7

    # A green/red passenger on a yellow-bordered platform.
    for y, x in disk:
        f1(7, ordinary, y, x)
    f1(7, 11, -2, 0)
    p["b1"][7] = -12

    def bridge_detector(out, dy, dx, capture):
        source_colors = ordinary if capture else (14, 9)
        middle_colors = ordinary if capture else pegs
        py, px = (1, 0) if dx else (0, 1)
        positive = dy > 0 or dx > 0

        sy = -dy if positive else -2 * dy
        sx = -dx if positive else -2 * dx
        my = 2 * dy if positive else dy
        mx = 2 * dx if positive else dx

        f1(out, source_colors, sy, sx)
        f1(out, source_colors, sy + py, sx + px)
        f1(out, source_colors, sy - py, sx - px, -1)
        f1(out, source_colors, sy + 2 * py, sx + 2 * px, -1)

        f1(out, middle_colors, my, mx, 0.5)
        f1(out, middle_colors, my + py, mx + px, 0.5)
        f1(out, middle_colors, my - py, mx - px, -0.5)
        f1(out, middle_colors, my + 2 * py, mx + 2 * px, -0.5)

        if not capture:
            if dx:
                f1(out, 15, my, mx)
            elif dy > 0:
                f1(out, 15, 2, -1)
            else:
                f1(out, 15, -2, -1)

        # The source and intervening object must be separated.
        f1(out, pegs, 0, 0, -3)
        f1(out, pegs, py, px, -3)
        p["b1"][out] = -2

    for i, (dy, dx) in enumerate(jump_dirs):
        bridge_detector(8 + i, dy, dx, False)
        bridge_detector(12 + i, dy, dx, True)

    def platform_edge(out, dy, dx, require_post):
        py, px = (0, 1) if dy else (1, 0)

        # Full yellow side plus corner and outward 2x2 rail.
        for t in (-1, 0, 1, 2):
            f1(out, (11, 15), t * py, t * px)
        f1(out, 11, -2 * py, -2 * px)

        for distance in (1, 2):
            for t in (0, 1):
                f1(
                    out, 5,
                    distance * dy + t * py,
                    distance * dx + t * px
                )

        p["b1"][out] = -8
        if require_post:
            f1(out, (7, 15), -2 * dy, -2 * dx)
            p["b1"][out] = -9

    for i, (dy, dx) in enumerate(arrow_dirs):
        platform_edge(16 + i, dy, dx, False)
        platform_edge(post_first[i], dy, dx, True)

    # Hinges of the rail graph's local degree.
    graph_colors = (5, 11, 12, 15, 7)
    for threshold in range(4):
        out = 20 + threshold
        f1(out, graph_colors, 0, 0, 8)
        for dy, dx in jump_dirs:
            f1(out, graph_colors, dy, dx)
        p["b1"][out] = -8 - threshold

    f1(25, 16, 0, 0)
    f1(26, 17, 0, 0)

    # Semantic features.
    f2(0, 0)
    f2(1, 1)
    f2(2, 30)                       # green-only centers
    for channel in (3, 4, 5):
        f2(3, channel)
    f2(4, 5)
    f2(5, 7)
    f2(6, 24)

    for i, (dy, dx) in enumerate(jump_dirs):
        for channel in (3, 4, 5):
            f2(7 + i, channel, -dy, -dx)
        f2(11 + i, 8 + i, dy, dx)
        f2(15 + i, 12 + i, dy, dx)

    for i in range(4):
        f2(19 + i, 16 + i)
        f2(preferred[i], post_first[i])

    # Coordinates restricted to landing rings.
    f2(26, 26)
    f2(26, 5, weight=2)
    p["b2"][26] = -2
    f2(27, 25)
    f2(27, 5, weight=2)
    p["b2"][27] = -2

    degree_coefficients = (
        (1, -1, -1, 1),
        (0, 0, 1, -2),
        (0, 0, 0, 1),
    )
    for out, coefficients in zip((28, 29, 30), degree_coefficients):
        for j, coefficient in enumerate(coefficients):
            f2(out, 20 + j, weight=coefficient)

    p["W3"][:, :32] = np.eye(32, dtype=np.float32)

    # Preferred-platform features:
    #   1 = feasible edge of a post platform;
    #  10 = feasible edge aligned with a loaded platform;
    #  11 = both.
    # The axial channels identify the row/column of the passenger.
    for i, (dy, dx) in enumerate(arrow_dirs):
        out = preferred[i]
        p["W3"][out].fill(0)
        p["W3"][out, 19 + i] = 20
        p["W3"][out, out] = 1
        p["W3"][out, (64 if dy else 32) + 5] = 10
        p["b3"][out] = -20

    # Empty-landing rays and gated source scores.
    for i, (dy, dx) in enumerate(jump_dirs):
        rec(i, i, dy, dx)
        p["Uh"][i, 3] = 12
        p["Uh"][i, 7 + i] = -60
        p["bh"][i] = -1

        rec(4 + i, i)
        p["Uh"][4 + i, 11 + i] = 60
        p["Uh"][4 + i, 15 + i] = 0.5
        p["bh"][4 + i] = -60.75

    # Attraction field: potential captures, forward boarding bridges,
    # and a weaker attraction to green pieces not already riding.
    for dy, dx in jump_dirs:
        rec(8, 8, dy, dx, 0.24)
    for i in range(4):
        p["Uh"][8, 15 + i] = 1
    p["Uh"][8, 12] += 1
    p["Uh"][8, 13] += 1
    p["Uh"][8, 2] += 0.02
    p["Uh"][8, 5] -= 0.02

    # Degree-normalized rail diffusion.
    graph_h = (9, 14, 15)
    for h, feature, degree in zip(graph_h, (28, 29, 30), (2, 3, 4)):
        for other in graph_h:
            for dy, dx in jump_dirs:
                rec(h, other, dy, dx, 0.995 / degree)
        rec(h, 8, weight=0.1)
        p["Uh"][h, feature] = 16
        p["bh"][h] = -16

    # Read the rail field just beyond each feasible platform edge.
    for i, (dy, dx) in enumerate(arrow_dirs):
        for h in graph_h:
            rec(10 + i, h, dy, dx)
        p["Uh"][10 + i, 19 + i] = 16
        p["bh"][10 + i] = -16

    # Click map: finish a selection, otherwise select a legal source.
    p["Wc"][0, 4] = 200
    p["Wc"][0, 26] = 0.4
    p["Wc"][0, 27] = 0.1
    for i, weight in enumerate((20, 24, 22, 18)):
        p["Wc"][0, 36 + i] = weight

    def hm(channel):
        return channel

    def fm(channel):
        return 32 + channel

    next_unit = 0

    def unit(coefficients, bias=0.0):
        nonlocal next_unit
        index = next_unit
        next_unit += 1
        for feature, weight in coefficients.items():
            p["Wz"][index, feature] += weight
        p["bz"][index] = bias
        return index

    def output(action, index, weight):
        p["Wo"][action, index] += weight

    def capped(action, coefficients, cap, weight, bias=0.0):
        lo = unit(coefficients, bias)
        hi = unit(coefficients, bias - cap)
        output(action, lo, weight)
        output(action, hi, -weight)

    p["bo"].fill(-1000)
    p["bo"][0] = -4
    p["bo"][6] = -5

    loaded_unit = unit({fm(5): 1})
    navigation_units = []

    for i, action in enumerate(actions):
        p["bo"][action] = -10

        legal_unit = unit({fm(19 + i): 1})
        output(action, legal_unit, 12)

        # If a piece is riding, do not choose a direction merely because
        # some unrelated, empty platform can move that way.
        output(action, loaded_unit, -24)
        capped(
            action, {fm(preferred[i]): 1},
            cap=1, weight=24, bias=-2
        )

        # Prefer post platforms before boarding. Once adjacent capturable
        # pieces exist, favor an empty ferry instead.
        capped(action, {fm(preferred[i]): 1}, 1, -1)
        post_coefficients = {
            fm(preferred[i]): 1,
            fm(5): -20,
        }
        for channel in range(15, 19):
            post_coefficients[fm(channel)] = -20
        capped(action, post_coefficients, 1, 2.5)

        nav_unit = unit({hm(10 + i): 1})
        navigation_units.append(nav_unit)

    ring_unit = unit({fm(4): 1})
    output(6, ring_unit, 40)

    # Captures in every direction.
    for h in range(4, 8):
        capped(6, {hm(h): 1}, 0.25, 120, -0.5)

    # Forward relocation jumps.
    for h in (5, 6):
        capped(6, {hm(h): 1}, 0.25, 120)

    # Backward rearrangements on boards containing blue pieces.
    for h in (4, 7):
        capped(
            6,
            {hm(h): 1, fm(1): 10, fm(5): -10},
            0.25, 120, -10
        )

    # Without a feasible platform move, allow any remaining legal jump.
    for h in range(4, 8):
        coefficients = {hm(h): 1}
        for i in range(4):
            coefficients[fm(19 + i)] = -10
        capped(6, coefficients, 0.25, 120)

    failure_unit = unit({fm(6): 1})
    output(0, failure_unit, 50)

    # Multiscale tanh readout approximates a logarithmic measure of
    # positive navigation signals, including distant, weak signals.
    for i, action in enumerate(actions):
        for k in range(13):
            memory_index = 13 * i + k
            p["Wmz"][memory_index, navigation_units[i]] = (
                10.0 ** (2 * k - 1)
            )
            p["Wo"][action, 128 + memory_index] = 1.0

    # Only mild heading persistence: rail gradients, not a fixed turn
    # sequence, determine the route.
    reverse = [1, 0, 3, 2]
    for i, action in enumerate(actions):
        memory_index = 52 + i
        p["Wma"][memory_index, action] = 3
        p["Wo"][action, 128 + memory_index] += 0.003
        p["Wo"][actions[reverse[i]], 128 + memory_index] -= 0.006

        memory_index = 56 + i
        p["Wm"][memory_index, memory_index] = 0.95
        p["Wma"][memory_index, action] = 0.1
        p["Wo"][action, 128 + memory_index] -= 0.004

    return p
