import math
import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,),
        "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,),
        "Wh": (16, 16, 3, 3),
        "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,),
        "Wm": (64, 64), "Wmz": (64, 128),
        "Wma": (64, 8), "bm": (64,),
        "Wo": (8, 192), "bo": (8,),
        "Wc": (1, 48), "bc": (1,),
    }
    p = {k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()}
    cardinal = ((-1, 0), (1, 0), (0, -1), (0, 1))
    square = tuple((y, x) for y in (-1, 0, 1)
                   for x in (-1, 0, 1))

    def c1(o, i, v=1.0, dy=0, dx=0):
        p["W1"][o, i, dy + 2, dx + 2] += v

    def c2(o, i, v=1.0, dy=0, dx=0):
        p["W2"][o, i, dy + 1, dx + 1] += v

    def f(o, i, v=1.0, pool=0):
        p["W3"][o, i + 32 * pool] += v

    def h(o, i, v=1.0, dy=0, dx=0):
        p["Wh"][o, i, dy + 1, dx + 1] += v

    def zmean(z, feature, v):
        p["Wz"][z, 64 + feature] += v

    def zmax(z, feature, v):
        p["Wz"][z, 32 + feature] += v

    def hmean(z, feature, v):
        p["Wz"][z, 16 + feature] += v

    # ------------------------------------------------------------
    # Layer one: eroded terrain, player interiors, marker detection.
    # ------------------------------------------------------------
    for dy, dx in square:
        for color in (5, 9, 10, 1):
            c1(0, color, dy=dy, dx=dx)
        c1(1, 10, dy=dy, dx=dx)
        for color in (5, 11):
            c1(4, color, dy=dy, dx=dx)
        for color in (5, 9, 10, 1, 11):
            c1(5, color, dy=dy, dx=dx)
    p["b1"][[0, 1, 4, 5]] = -8.0

    # Selected marker: narrow colour-11 interior.
    c1(2, 11)
    for dy, dx in cardinal:
        c1(2, 11, -1.0, 2 * dy, 2 * dx)

    c1(3, 9)
    c1(6, 1)

    # Horizontal and vertical coordinate hinges.
    c1(7, 17, 63.0)
    p["b1"][7] = -31.0
    c1(8, 17, -63.0)
    p["b1"][8] = 31.0
    c1(9, 16, 63.0)
    p["b1"][9] = -31.0

    colors = (12, 14, 15)
    for j, color in enumerate(colors):
        # A clipped dilation represented by two ReLU channels.
        for dy, dx in square:
            c1(10 + j, color, dy=dy, dx=dx)
            c1(13 + j, color, dy=dy, dx=dx)
        p["b1"][13 + j] = -1.0

        # Long-run evidence distinguishes gates from single cells.
        for d in range(-2, 3):
            c1(16 + j, color, dx=d)
            c1(19 + j, color, dy=d)
        p["b1"][16 + j] = -4.0
        p["b1"][19 + j] = -4.0

        # Exterior samples identify colours used for wall bands.
        c1(22 + j, color)
        c1(22 + j, 18, -1.0, dx=-2)
        c1(22 + j, 18, -1.0, dx=2)
        p["b1"][22 + j] = 1.0

    # Blocker interior close to a player.
    c1(25, 9, 25.0)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            c1(25, 10, dy=dy, dx=dx)
    p["b1"][25] = -25.0

    c1(26, 16)

    # One corner per visible controlled player rectangle.
    c1(27, 10)
    c1(27, 10, -1.0, dy=-1)
    c1(27, 10, -1.0, dx=-1)

    for j, (dy, dx) in enumerate(cardinal):
        for color in (5, 9, 10, 1, 11):
            c1(28 + j, color, dy=2 * dy, dx=2 * dx)

    # ------------------------------------------------------------
    # Layer two: directional edges, coordinate moments, parking.
    # ------------------------------------------------------------
    c2(0, 0)
    c2(1, 2)
    c2(2, 3)

    for obj in (1, 2):
        c2(3, obj)
        c2(30, obj)
        c2(31, obj)

    c2(3, 7, 0.001)
    c2(3, 8, 0.001)
    p["b2"][3] = -1.0

    c2(30, 26, 0.01)
    p["b2"][30] = -1.0

    c2(31, 9, 0.002)
    c2(31, 26, -0.063)
    p["b2"][31] = -0.969

    c2(4, 25)
    c2(5, 6, 100.0)
    c2(5, 27)
    c2(6, 4)

    # Dead-end parking candidates.
    c2(7, 5, 4.0)
    for j, (dy, dx) in enumerate(cardinal):
        c2(7, 28 + j, -1.0, dy, dx)
    p["b2"][7] = -2.0

    # Rendezvous seeds on the central column.
    c2(8, 0)
    c2(8, 7, -1.0)
    c2(8, 8, -1.0)

    for out, (dy, dx) in zip(range(9, 13), cardinal):
        for obj in (1, 2):
            c2(out, obj)
            c2(out, obj, -1.0, dy, dx)

    c2(13, 7, -1.0)
    p["b2"][13] = 1.0
    c2(14, 9, -1.0)
    p["b2"][14] = 1.0

    for j in range(3):
        c2(15 + j, 10 + j)
        c2(18 + j, 13 + j)

        for d in (-1, 0, 1):
            c2(21 + j, 16 + j, dy=d)
            c2(24 + j, 19 + j, dx=d)
        c2(27 + j, 22 + j)

    # ------------------------------------------------------------
    # Axial layer: palette suppression and spatial-half separation.
    # ------------------------------------------------------------
    f(0, 0)
    for j in range(3):
        # Relaxed terrain includes coloured gates.
        f(1 + j, 15 + j)
        f(1 + j, 18 + j, -1.0)
        f(1 + j, 27 + j, -1.0, pool=1)

        # Switch guidance excludes closed gates.
        f(24 + j, 15 + j)
        f(24 + j, 18 + j, -1.0)
        f(24 + j, 27 + j, -1.0, pool=1)
        f(24 + j, 21 + j, -4.0, pool=1)
        f(24 + j, 24 + j, -4.0, pool=2)

    f(4, 8)
    f(5, 2)
    f(6, 1)

    f(7, 5)
    p["b3"][7] = -1.0

    f(8, 7)
    f(9, 6)

    edge_specs = (
        (9, 14, True), (10, 14, True),
        (9, 14, False), (10, 14, False),
        (11, 13, True), (12, 13, True),
        (11, 13, False), (12, 13, False),
    )
    for out, (edge, half, positive) in zip(range(10, 18), edge_specs):
        f(out, edge)
        f(out, half, 1.0 if positive else -1.0)
        if positive:
            p["b3"][out] = -1.0

    f(18, 3)
    f(19, 30)
    f(20, 31)
    f(21, 5)
    f(22, 4)

    f(23, 7, 0.5)
    f(23, 1)
    p["b3"][23] = -1.0

    # ------------------------------------------------------------
    # Tied block: three saturated wavefronts.
    #
    # 0/1: rendezvous reachability
    # 2/3: accessible-switch reachability
    # 4: combined arrival-time potential
    # 5/6: parking reachability
    # 7: parking arrival-time potential
    # 8..15: positive directional improvements
    # ------------------------------------------------------------
    neighborhood = ((0, 0),) + cardinal
    for first, kind in ((0, "primary"), (2, "switch"), (5, "parking")):
        for out in (first, first + 1):
            for dy, dx in neighborhood:
                h(out, first, 1.0, dy, dx)
                h(out, first + 1, -1.0, dy, dx)

            if kind == "primary":
                for terrain in (0, 1, 2, 3):
                    p["Uh"][out, terrain] += 16.0
                p["Uh"][out, 4] += 1.0
            elif kind == "switch":
                for terrain in (0, 24, 25, 26):
                    p["Uh"][out, terrain] += 16.0
                for seed in (24, 25, 26):
                    p["Uh"][out, seed] += 1.0
            else:
                p["Uh"][out, 9] += 16.0
                p["Uh"][out, 8] += 2.0

            p["bh"][out] = -16.0 - float(out == first + 1)

    h(4, 4)
    h(4, 0)
    h(4, 1, -1.0)
    h(4, 2, 2.5)
    h(4, 3, -2.5)

    h(7, 7)
    h(7, 5)
    h(7, 6, -1.0)

    read_directions = (
        (-1, 0), (1, 0), (-1, 0), (1, 0),
        (0, -1), (0, 1), (0, -1), (0, 1),
    )
    for out, mask, (dy, dx) in zip(
            range(8, 16), range(10, 18), read_directions):
        for potential in (4, 7):
            h(out, potential, 1.0, dy, dx)
            h(out, potential, -1.0)
        p["Uh"][out, mask] = 512.0
        p["bh"][out] = -512.0

    # ------------------------------------------------------------
    # Global features.
    # ------------------------------------------------------------
    groups = (
        (13, 14), (12, 15),       # horizontal inward / outward
        (8, 10), (9, 11),         # physical up / down
        (9, 10), (8, 11),         # vertical inward / outward
        (12, 14), (13, 15),       # physical left / right
    )
    for z, group in enumerate(groups):
        for channel in group:
            hmean(z, channel, 4096.0 / 3.0)

    for z, feature in ((8, 18), (9, 19), (10, 20)):
        zmean(z, feature, 128.0)

    zmean(11, 21, 4096.0)
    zmean(12, 21, 4096.0)
    p["bz"][11] = -2.0
    p["bz"][12] = -3.0

    zmax(13, 6, 1.0)

    zmax(14, 22, 1.0)
    zmax(15, 22, 1.0)
    p["bz"][14] = -0.5
    p["bz"][15] = -1.5

    zmax(16, 23, 1.0)

    # ------------------------------------------------------------
    # Motion-based control-reflection estimates.
    # ------------------------------------------------------------
    for m, z in enumerate((8, 9, 10)):
        p["Wmz"][m, z] = 1.0

    def reflection(first, metric_m, metric_z, gain,
                   action_a, sign_a, action_b, sign_b):
        u, v, baseline, orient = range(first, first + 4)
        gate = 2000.0
        for unit, action, sign in (
                (u, action_a, sign_a), (v, action_b, sign_b)):
            p["Wmz"][unit, metric_z] = gain * sign
            p["Wm"][unit, metric_m] = -gain * sign
            p["Wma"][unit, action] = gate
            p["Wmz"][unit, 13] = -gate
            p["bm"][unit] = -gate

        p["Wma"][baseline, action_a] = gate
        p["Wma"][baseline, action_b] = gate
        p["Wmz"][baseline, 13] = -gate
        p["bm"][baseline] = -gate

        p["Wm"][orient, orient] = 2.0
        p["Wm"][orient, u] = 4.0
        p["Wm"][orient, v] = 4.0
        p["Wm"][orient, baseline] = -4.0
        p["bm"][orient] = 4.0
        return orient

    ox = reflection(3, 0, 8, 3000.0, 4, -1.0, 3, 1.0)
    oy = reflection(7, 1, 9, 20000.0, 2, 1.0, 1, -1.0)
    om = reflection(11, 2, 10, 3000.0, 2, -1.0, 1, 1.0)

    next_mem = 15

    def route(orientation, sign, mode, assignments):
        nonlocal next_mem
        baseline = next_mem
        next_mem += 1

        def configure(unit, score=None):
            p["Wm"][unit, orientation] = 8.0 * sign
            p["bm"][unit] = -8.0
            p["Wmz"][unit, 13] = -30.0

            if mode == "two":
                p["Wmz"][unit, 11] -= 20.0
                p["Wmz"][unit, 12] += 20.0
            elif mode == "four":
                p["Wmz"][unit, 11] += 20.0
                p["Wmz"][unit, 12] -= 20.0
                p["bm"][unit] -= 20.0

            if score is not None:
                p["Wmz"][unit, score] += 0.3

        configure(baseline)
        for score, action in assignments:
            unit = next_mem
            next_mem += 1
            configure(unit, score)
            p["Wo"][action, 128 + unit] += 6.0
            p["Wo"][action, 128 + baseline] -= 6.0

    route(ox, 1.0, None, ((0, 4), (1, 3)))
    route(ox, -1.0, None, ((0, 3), (1, 4)))
    route(oy, 1.0, "two", ((2, 1), (3, 2)))
    route(oy, -1.0, "two", ((2, 2), (3, 1)))
    route(om, 1.0, "four", ((4, 2), (5, 1)))
    route(om, -1.0, "four", ((4, 1), (5, 2)))

    # Selected blockers use physical controls.
    baseline = 33
    for unit in range(33, 38):
        p["Wmz"][unit, 13] = 30.0
        p["bm"][unit] = -30.0

    for unit, score, action in (
            (34, 6, 3), (35, 7, 4), (36, 2, 1), (37, 3, 2)):
        p["Wmz"][unit, score] = 0.3
        p["Wo"][action, 128 + unit] += 6.0
        p["Wo"][action, 128 + baseline] -= 6.0

    # Small deterministic exploration signal.
    angle = 0.73
    co, si = math.cos(angle), math.sin(angle)
    p["Wm"][38, 38] = 1.25 * co
    p["Wm"][38, 39] = -1.25 * si
    p["Wm"][39, 38] = 1.25 * si
    p["Wm"][39, 39] = 1.25 * co
    p["bm"][38] = 0.13

    p["Wo"][1, 128 + 38] = 0.07
    p["Wo"][2, 128 + 38] = -0.07
    p["Wo"][3, 128 + 39] = 0.07
    p["Wo"][4, 128 + 39] = -0.07

    p["bo"][:] = -100.0
    p["bo"][1:5] = 0.0
    p["bo"][1] = 0.08

    # ------------------------------------------------------------
    # No-effect penalty based on an observed-position fingerprint.
    # ------------------------------------------------------------
    fingerprint = ((8, 0, 1.0), (9, 1, 4.0), (10, 2, 0.37))
    for action in range(1, 5):
        u = 40 + 2 * (action - 1)
        v = u + 1
        previous = 48 + action - 1

        for unit, sign in ((u, 1.0), (v, -1.0)):
            for z, m, weight in fingerprint:
                p["Wmz"][unit, z] = sign * 10000.0 * weight
                p["Wm"][unit, m] = -sign * 10000.0 * weight
            p["Wma"][unit, action] = 2000.0
            p["bm"][unit] = -2003.0

        p["Wma"][previous, action] = 5.0
        penalty = 12.0
        p["Wo"][action, 128 + u] += penalty
        p["Wo"][action, 128 + v] += penalty
        p["Wo"][action, 128 + previous] += -2.0 * penalty + 0.10
        p["bo"][action] += 2.0 * penalty

    # ------------------------------------------------------------
    # Reversal feedback. Repeated reversals temporarily discourage
    # their axis; a straight run does not accumulate this penalty.
    # ------------------------------------------------------------
    constant = 57
    p["bm"][constant] = 10.0

    def reversal_feedback(direction, test_a, test_b, accumulator,
                          action_a, action_b, strength):
        p["Wma"][direction, action_a] = 5.0
        p["Wma"][direction, action_b] = -5.0

        p["Wm"][test_a, direction] = -4.0
        p["Wma"][test_a, action_a] = 8.0
        p["bm"][test_a] = -10.0

        p["Wm"][test_b, direction] = 4.0
        p["Wma"][test_b, action_b] = 8.0
        p["bm"][test_b] = -10.0

        p["Wm"][accumulator, accumulator] = 0.85
        p["Wm"][accumulator, test_a] = 0.75
        p["Wm"][accumulator, test_b] = 0.75
        p["Wm"][accumulator, constant] = 1.5

        p["Wo"][action_a, 128 + accumulator] -= strength
        p["Wo"][action_b, 128 + accumulator] -= strength

    reversal_feedback(53, 54, 55, 56, 3, 4, 3.0)
    reversal_feedback(58, 59, 60, 61, 1, 2, 2.4)

    # ------------------------------------------------------------
    # Blocker selection, parking completion, timeout, and click map.
    # ------------------------------------------------------------
    timer = 52
    p["Wm"][timer, timer] = 0.98
    p["Wmz"][timer, 13] = 5.05
    p["bm"][timer] = -5.0

    p["bo"][6] = -5.0
    p["Wo"][6, 14] = 40.0
    p["Wo"][6, 15] = -40.0
    p["Wo"][6, 13] = -2.0
    p["Wo"][6, 16] = 50.0
    p["Wo"][6, 128 + timer] = 20.0

    # Nearby blocker interiors win in collective mode. Inactive
    # player pixels win in selected mode, restoring collective play.
    p["Wc"][0, 5] = 100.0
    p["Wc"][0, 22] = 2.0
    p["Wc"][0, 7] = 10.0

    return p
