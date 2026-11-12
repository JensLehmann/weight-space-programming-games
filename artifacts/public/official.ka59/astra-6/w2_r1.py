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

    def w1(out, inp, dy=0, dx=0, value=1.0):
        p["W1"][out, inp, dy + 2, dx + 2] += value

    def colors(out, indices, dy=0, dx=0, value=1.0):
        for color in indices:
            w1(out, color, dy, dx, value)

    def w2(out, inp, dy=0, dx=0, value=1.0):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    def w3(out, inp, value=1.0):
        p["W3"][out, inp] += value

    def wh(out, inp, dy=0, dx=0, value=1.0):
        p["Wh"][out, inp, dy + 1, dx + 1] += value

    # ------------------------------------------------------------
    # Local perception.
    # ------------------------------------------------------------

    # Small-body footprint obstacles.
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            w1(0, 2, dy, dx)
            colors(1, (2, 15), dy, dx)

    w1(2, 14)
    w1(5, 11)

    # Obstacle cross-sections three pixels ahead after W2 shifts.
    directional_obstacles = [3, 4, 6, 31]
    for out, (dy, dx) in zip(directional_obstacles, directions):
        py, px = -dx, dy
        for lateral in (-1, 0, 1):
            colors(
                out, (2, 15),
                2 * dy + lateral * py,
                2 * dx + lateral * px,
            )

    # Selected small core: black center and at least three cardinal
    # yellow neighbors. This rejects most black contact-edge pixels.
    for out, bias in ((7, -10.0), (8, -11.0)):
        w1(out, 0, value=8.0)
        for dy, dx in directions:
            w1(out, 14, dy, dx)
        p["b1"][out] = bias

    tall_core = [(0, 0), (1, 0)]
    tall_sides = [
        (-1, 0), (2, 0),
        (0, -1), (0, 1),
        (1, -1), (1, 1),
    ]
    wide_core = [(0, 0), (0, 1)]
    wide_sides = [(dx, dy) for dy, dx in tall_sides]
    square_core = [(0, 0), (0, 1), (1, 0), (1, 1)]
    square_sides = [(-1, 0), (0, -1)]

    # First pixel of a larger selected core.
    for out, core, sides in (
        (9, tall_core, tall_sides),
        (10, wide_core, wide_sides),
        (11, square_core, square_sides),
    ):
        for dy, dx in core:
            w1(out, 0, dy, dx, 4.0)
        for dy, dx in sides:
            w1(out, 14, dy, dx)
        p["b1"][out] = -(4 * len(core) + len(sides) - 1)

    # Inactive controllers.
    colors(12, (4, 5))
    for dy, dx in directions:
        w1(12, 14, dy, dx)
    p["b1"][12] = -4.0

    for out, core, sides in (
        (13, tall_core, tall_sides),
        (14, wide_core, wide_sides),
        (15, square_core, square_sides),
    ):
        for dy, dx in core:
            colors(out, (4, 5), dy, dx, 4.0)
        for dy, dx in sides:
            w1(out, 14, dy, dx)
        p["b1"][out] = -(4 * len(core) + len(sides) - 1)

    # Occlusion-tolerant small goal.
    #
    # Every border position must contain either the outline or a
    # movable piece, and at least eight border pixels must be outline.
    # Channels 16 and 29 form an exactly clipped indicator.
    border = [
        (dy, dx)
        for dy in range(-2, 3)
        for dx in range(-2, 3)
        if abs(dy) == 2 or abs(dx) == 2
    ]
    border_colors = (0, 4, 5, 11, 12, 13, 14)
    for out, extra in ((16, 0.0), (29, 1.0)):
        for dy, dx in border:
            colors(out, border_colors, dy, dx, 16.0)
            w1(out, 4, dy, dx)
        p["b1"][out] = -(16.0 * len(border) + 7.0 + extra)

    # Larger-goal corner and edge evidence.
    for dy, dx in [
        (-1, -1), (-1, 0), (0, -1), (-1, 2), (2, -1)
    ]:
        w1(17, 4, dy, dx)
    for dy, dx in [(-2, -1), (-1, -2)]:
        w1(17, 4, dy, dx, -2.0)
    p["b1"][17] = -4.0
    w1(18, 4, -1, 2)
    w1(19, 4, 2, -1)

    # Cross goal center.
    for dy, dx in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
        w1(20, 4, dy, dx)
    for dy, dx in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        w1(20, 4, dy, dx, -4.0)
    p["b1"][20] = -3.0

    # Precisely aligned approach positions for small controllers.
    # W2 shifts these predicates by one more pixel: the tested core
    # then lies three pixels ahead of the approach center.
    for d, (dy, dx) in enumerate(directions):
        out = 21 + d
        py, px = -dx, dy
        colors(out, (4, 5), 2 * dy, 2 * dx, 4.0)
        w1(out, 14, 2 * dy + py, 2 * dx + px)
        w1(out, 14, 2 * dy - py, 2 * dx - px)
        p["b1"][out] = -5.0

    # Movable occupancy, excluding the selected body's neighborhood.
    colors(25, (5, 11, 12, 13, 14))
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            w1(25, 0, dy, dx, -1.0)

    w1(26, 16)
    w1(26, 17, value=1.0 / 128.0)

    for out in (27, 28):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                w1(out, 11, dy, dx)
    p["b1"][28] = -1.0

    # A scene moment distinguishes movement and successful pushes
    # from actions that leave the scene unchanged.
    colors(30, (0, 4, 5, 11, 12, 13, 14), value=2.0)
    w1(30, 16)
    w1(30, 17, value=1.0 / 128.0)
    p["b1"][30] = -2.0

    # ------------------------------------------------------------
    # Second layer: anchors, goals, and approach masks.
    # ------------------------------------------------------------

    small_terms = [(7, 0, 0, 1.0), (8, 0, 0, -1.0)]
    large_terms = [
        (9, 1, 0, 1.0),
        (10, 0, 1, 1.0),
        (11, 1, 1, 1.0),
    ]
    inactive_terms = [
        (12, 0, 0, 1.0),
        (13, 1, 0, 1.0),
        (14, 0, 1, 1.0),
        (15, 1, 1, 1.0),
    ]

    def terms(out, entries, scale=1.0):
        for src, dy, dx, value in entries:
            w2(out, src, dy, dx, scale * value)

    w2(0, 0)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            w2(1, 0, dy, dx)

    w2(2, 1)
    w2(3, 1)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            w2(3, 25, dy, dx)

    # Remove larger-core detections from the small-core channel.
    terms(4, small_terms)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            for src in (9, 10, 11):
                w2(4, src, dy, dx, -1.0)

    terms(5, inactive_terms)
    terms(6, small_terms)
    terms(7, large_terms)

    # Empty small goal.
    w2(8, 16)
    w2(8, 29, value=-1.0)
    terms(8, small_terms, -1.0)
    w2(8, 12, value=-1.0)

    def larger_goal(out, kind, occupied_sign):
        w2(out, 17, -1, -1)
        if kind == "tall":
            w2(out, 19, 1, -1)
            w2(out, 18, -1, 1, -1.0)
            p["b2"][out] = -1.0
            bodies = [(9, 1, 0), (13, 1, 0)]
        elif kind == "wide":
            w2(out, 18, -1, 1)
            w2(out, 19, 1, -1, -1.0)
            p["b2"][out] = -1.0
            bodies = [(10, 0, 1), (14, 0, 1)]
        else:
            w2(out, 18, -1, 1)
            w2(out, 19, 1, -1)
            p["b2"][out] = -2.0
            bodies = [(11, 1, 1), (15, 1, 1)]

        for src, dy, dx in bodies:
            w2(out, src, dy, dx, occupied_sign)
        if occupied_sign > 0:
            p["b2"][out] -= 1.0

    larger_goal(9, "tall", -1.0)
    larger_goal(10, "wide", -1.0)
    larger_goal(11, "square", -1.0)

    w2(12, 20)
    w2(12, 5, value=-1.0)

    for d, (dy, dx) in enumerate(directions):
        w2(13 + d, 21 + d, dy, dx)
        w2(13 + d, 1, value=-1.0)
        w2(13 + d, 25, value=-1.0)

        w2(17 + d, directional_obstacles[d], dy, dx)

    # Completed small goal.
    w2(21, 16)
    w2(21, 29, value=-1.0)
    terms(21, small_terms)
    w2(21, 12)
    p["b2"][21] = -1.0

    larger_goal(28, "tall", 1.0)
    larger_goal(29, "wide", 1.0)
    larger_goal(30, "square", 1.0)

    w2(22, 30)

    for out in (23, 24):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                w2(out, 27, dy, dx)
                w2(out, 28, dy, dx, -1.0)
    p["b2"][24] = -1.0

    w2(25, 2)
    terms(25, small_terms, -1.0)
    terms(25, large_terms, -1.0)

    w2(26, 26)
    w2(27, 26, value=200.0)
    p["b2"][27] = -199.2
    p["b2"][31] = 1.0

    # ------------------------------------------------------------
    # Static recurrent inputs.
    # ------------------------------------------------------------

    # 0 normal obstacles, 1 solid-only obstacles, 3 navigation obstacles.
    for out, src in [(0, 2), (1, 0), (2, 1), (3, 3)]:
        w3(out, src)
        w3(out, 27)

    w3(4, 4)
    w3(4, 7)
    w3(5, 5)

    for src in (8, 9, 10, 11):
        w3(6, src)
    w3(7, 12)

    for d in range(4):
        w3(8 + d, 13 + d)
        w3(8 + d, 3, -1.0)
        w3(8 + d, 25, -1.0)
        w3(8 + d, 27, -1.0)

        w3(12 + d, 4)
        w3(12 + d, 7)
        w3(12 + d, 17 + d, -1.0)
        w3(12 + d, 27, -1.0)

    completed = (21, 28, 29, 30)

    w3(16, 5)
    for src in completed:
        w3(16, src, -1.0)

    w3(17, 4)
    w3(17, 7)
    for src in completed:
        w3(17, src)
    p["b3"][17] = -1.0

    w3(18, 23)
    w3(18, 24, -1.0)

    w3(19, 26)
    w3(19, 4, 2.0)
    w3(19, 7, 2.0)
    p["b3"][19] = -2.0

    w3(20, 4)
    w3(20, 7)
    w3(20, 5)
    w3(21, 31)

    w3(23, 22)
    w3(23, 27, -1000.0)

    # ------------------------------------------------------------
    # Recurrent computation.
    #
    # 0,1: capped ordinary-reachability wave from all controllers
    # 2: iteration clock
    # 3: delayed disconnected-goal seeds
    # 4,5: capped goal wave, allowed to cross gulches
    # 6..9: integrated directional goal-arrival differences
    # 10,11: capped navigation wave
    # 12..15: action arrival scores at the selected anchor
    #
    # A capped value is represented by u - ReLU(u - 1).
    # ------------------------------------------------------------

    gate = 32768.0

    for out in (0, 1):
        for dy, dx in directions:
            wh(out, 0, dy, dx)
            wh(out, 1, dy, dx, -1.0)
        p["Uh"][out, 20] = 1.0
        p["Uh"][out, 0] = -gate
    p["bh"][1] = -1.0

    wh(2, 2)
    p["Uh"][2, 21] = 1.0

    # Only goals still disconnected after the initial reachability
    # expansion inject the launch-planning wave.
    wh(3, 2)
    wh(3, 0, value=-1024.0)
    wh(3, 1, value=1024.0)
    p["Uh"][3, 6] = 1024.0
    p["bh"][3] = -1044.0

    for out in (4, 5):
        for dy, dx in directions:
            wh(out, 4, dy, dx)
            wh(out, 5, dy, dx, -1.0)
        wh(out, 3)
        p["Uh"][out, 1] = -gate
    p["bh"][5] = -1.0

    # Integrating the difference between two binary arrivals gives
    # a distance-gradient estimate without diffusion attenuation.
    for d, (dy, dx) in enumerate(directions):
        out = 6 + d
        wh(out, out)
        wh(out, 4, dy, dx)
        wh(out, 5, dy, dx, -1.0)
        wh(out, 4, -dy, -dx, -1.0)
        wh(out, 5, -dy, -dx)
        p["Uh"][out, 8 + d] = 1024.0
        p["bh"][out] = -1024.0

    for out in (10, 11):
        for dy, dx in directions:
            wh(out, 10, dy, dx)
            wh(out, 11, dy, dx, -1.0)
        for src in range(6, 10):
            wh(out, src)

        # Weak ordinary seeds grow into a wave later than nearby
        # launch sources. This gives transport precedence over parking.
        p["Uh"][out, 6] = np.float32(1.0e-35)
        p["Uh"][out, 3] = -gate
    p["bh"][11] = -1.0

    for d, (dy, dx) in enumerate(directions):
        out = 12 + d
        wh(out, out)
        wh(out, 10, dy, dx)
        wh(out, 11, dy, dx, -1.0)

        # At an approach position, perform the indicated push rather
        # than orbiting the navigation source.
        wh(out, 6 + d, value=0.25)

        p["Uh"][out, 12 + d] = gate
        p["bh"][out] = -gate

    # ------------------------------------------------------------
    # Readout.
    # ------------------------------------------------------------

    for d in range(4):
        p["Wz"][d, 12 + d] = 1.0
        p["Wo"][1 + d, d] = 1.0

    p["Wz"][4, 32 + 5] = 1.0
    p["Wz"][5, 32 + 16] = 1.0
    p["Wz"][6, 32 + 17] = 1.0

    p["Wz"][7, 32 + 16] = 1.0
    p["Wz"][7, 32 + 17] = 1.0
    p["bz"][7] = -1.0

    p["Wz"][8, 32 + 19] = 1.0
    p["Wz"][8, 64 + 23] = 32.0

    # ------------------------------------------------------------
    # Bounded stagnation memory.
    # ------------------------------------------------------------

    p["Wmz"][0, 8] = 1.0e-4

    slope = 100000.0
    exclusion = 1048576.0
    tolerance = 1.0e-5

    for d in range(4):
        action = 1 + d
        plus = 1 + 2 * d
        minus = plus + 1

        for unit, sign in ((plus, 1.0), (minus, -1.0)):
            p["Wmz"][unit, 8] = slope
            p["Wm"][unit, 0] = -10000.0 * slope
            p["Wma"][unit, action] = exclusion
            p["bm"][unit] = sign * slope * tolerance - exclusion

        p["Wo"][action, 128 + plus] = -120.0
        p["Wo"][action, 128 + minus] = 120.0

        p["Wma"][9 + d, action] = 2.0
        p["Wo"][action, 128 + 9 + d] = 0.001

    p["bo"][0] = -10000.0
    p["bo"][5] = -10000.0
    p["bo"][7] = -10000.0

    # Click if no useful action wave reaches the selected controller,
    # or hand off a completed controller to an unfinished one.
    p["bo"][6] = -1.0
    p["Wo"][6, 4] = 1.01
    p["Wo"][6, 7] = 1000.0

    p["Wc"][0, 5] = 10000.0
    p["Wc"][0, 16] = 100000.0
    p["Wc"][0, 19] = -1.0

    return p
