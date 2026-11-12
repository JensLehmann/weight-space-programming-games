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
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def w1(out, inp, value, dy=0, dx=0):
        p["W1"][out, inp, 2 + dy, 2 + dx] += value

    def w2(out, inp, value, dy=0, dx=0):
        p["W2"][out, inp, 1 + dy, 1 + dx] += value

    def wh(out, inp, value, dy=0, dx=0):
        p["Wh"][out, inp, 1 + dy, 1 + dx] += value

    def player(out, value, dy=0, dx=0):
        w1(out, 10, value, dy, dx)
        w1(out, 1, value, dy, dx)

    # Visual primitives.
    w1(0, 5, 1.0)
    player(1, 1.0)
    w1(2, 9, 1.0)
    w1(3, 11, 1.0)

    # Column extrema restricted to player pixels.
    w1(4, 17, 1.0)
    player(4, 2.0)
    p["b1"][4] = -2.0

    w1(5, 17, -1.0)
    player(5, 2.0)
    p["b1"][5] = -1.0

    # Clipped left/right membership.
    for out, bias in ((6, -65.0), (7, -66.0)):
        w1(out, 17, -126.0)
        player(out, 128.0)
        p["b1"][out] = bias

    for out, bias in ((28, -191.0), (29, -192.0)):
        w1(out, 17, 126.0)
        player(out, 128.0)
        p["b1"][out] = bias

    # Up/down/left/right player edges.
    for d, (dy, dx) in enumerate(directions):
        player(8 + d, 1.0)
        player(8 + d, -1.0, dy, dx)

    # Distinguish isolated switch patches from long gate segments.
    for c, colour in enumerate((12, 14, 15)):
        for d, (dy, dx) in enumerate(directions):
            w1(12 + 4 * c + d, colour, 1.0, 2 * dy, 2 * dx)
        w1(24 + c, colour, 1.0)

    w1(27, 1, 1.0)
    w1(30, 17, 1.0)
    w1(31, 16, 1.0)

    # Second visual layer.
    for i in (0, 1, 2, 3, 4, 5, 8, 9, 10, 11):
        w2(i, i, 1.0)

    w2(6, 6, 1.0)
    w2(6, 7, -1.0)
    w2(7, 28, 1.0)
    w2(7, 29, -1.0)

    for c in range(3):
        base = 12 + 4 * c
        horizontal = 12 + 2 * c
        vertical = horizontal + 1

        w2(horizontal, base + 2, 1.0, 0, -1)
        w2(horizontal, base + 3, 1.0, 0, 1)
        p["b2"][horizontal] = -1.0

        w2(vertical, base, 1.0, -1, 0)
        w2(vertical, base + 1, 1.0, 1, 0)
        p["b2"][vertical] = -1.0

        w2(18 + c, 24 + c, 1.0)

    w2(21, 27, 1.0)
    w2(22, 30, 1.0)
    w2(23, 31, 1.0)

    # Row extrema restricted to players.
    w2(24, 31, 1.0)
    w2(24, 1, 1.0)
    p["b2"][24] = -1.0

    w2(25, 31, -1.0)
    w2(25, 1, 1.0)

    # Axial features.
    p["W3"][0, 0] = 1.0
    p["W3"][0, 1] = 1.0

    for c in range(3):
        p["W3"][1 + c, 18 + c] = 1.0
        p["W3"][1 + c, 32 + 12 + 2 * c] = -1.0
        p["W3"][1 + c, 64 + 13 + 2 * c] = -1.0

    p["W3"][4, 6] = 1.0
    p["W3"][5, 7] = 1.0

    for half in range(2):
        for d in range(4):
            out = 6 + 4 * half + d
            p["W3"][out, 8 + d] = 1.0
            p["W3"][out, 6 + half] = 1.0
            p["b3"][out] = -1.0

    for out, inp in {
        14: 4,
        15: 5,
        16: 21,
        17: 2,
        18: 3,
        19: 24,
        20: 25,
        26: 23,
    }.items():
        p["W3"][out, inp] = 1.0

    # Stable diffusion fields.
    #
    # Each field averages four neighbours and injects 0.001 at
    # its sources. Free pixels have zero drive apart from sources;
    # obstacles have drive -1 and therefore stay at zero.
    # Starting from zero, both fields are bounded above by 0.128.
    for field, source in ((3, 4), (7, 5)):
        for dy, dx in directions:
            wh(field, field, 0.25, dy, dx)
        for traversable in range(4):
            p["Uh"][field, traversable] = 1.0
        p["Uh"][field, source] += 0.001
        p["bh"][field] = -1.0

    # Positive gradient of the opposite-half field, gated to
    # player boundaries. Gate drive is exactly zero where active,
    # avoiding cancellation of very small navigation signals.
    for half, opposite_field in ((0, 7), (1, 3)):
        for d, (dy, dx) in enumerate(directions):
            out = 8 + 4 * half + d
            wh(out, opposite_field, 1.0, dy, dx)
            wh(out, opposite_field, -1.0)
            p["Uh"][out, 6 + 4 * half + d] = 1.0
            p["bh"][out] = -1.0

    # Candidate actions expressed geometrically:
    # up, down, outward horizontally, inward horizontally.
    candidates = [
        (8, 12),
        (9, 13),
        (10, 15),
        (11, 14),
    ]
    for i, channels in enumerate(candidates):
        for channel in channels:
            p["Wz"][i, channel] = 1.0

    # Pairwise regret features, without adding a large baseline.
    regret_units = {}
    next_unit = 4
    for i in range(4):
        regret_units[i] = []
        for j in range(4):
            if i == j:
                continue
            u = next_unit
            next_unit += 1
            regret_units[i].append(u)
            for channel in candidates[j]:
                p["Wz"][u, channel] += 1.0
            for channel in candidates[i]:
                p["Wz"][u, channel] -= 1.0

    # Coordinate extrema and selection flags.
    for z_index, f_channel in (
        (32, 14),
        (33, 15),
        (34, 19),
        (35, 20),
        (36, 16),
        (37, 17),
    ):
        p["Wz"][z_index, 32 + f_channel] = 1.0

    # Memory constant and previous coordinate summaries.
    p["bm"][0] = 20.0
    for d in range(4):
        p["Wmz"][2 + d, 32 + d] = 0.01

    # Infer horizontal handedness from changes in player span.
    evidence = [
        (6, -1.0, 4),
        (7, 1.0, 3),
        (8, 1.0, 4),
        (9, -1.0, 3),
    ]
    for unit, sign, action in evidence:
        for z_index in (32, 33):
            p["Wmz"][unit, z_index] = 100.0 * sign
        for mem_index in (2, 3):
            p["Wm"][unit, mem_index] = -10000.0 * sign
        p["Wma"][unit, action] = 1000.0
        p["bm"][unit] = -1002.0

    p["Wm"][1, 1] = 20.0
    p["Wm"][1, 0] = -10.0
    p["bm"][1] = 10.0
    p["Wm"][1, 6:8] = 30.0
    p["Wm"][1, 8:10] = -30.0

    # Movement detectors.
    for d in range(4):
        for k, sign in enumerate((1.0, -1.0)):
            unit = 10 + 2 * d + k
            p["Wmz"][unit, 32 + d] = 1000.0 * sign
            p["Wm"][unit, 2 + d] = -100000.0 * sign
            p["bm"][unit] = -15.0

    # Stagnation increases only after repeated lack of movement.
    p["Wm"][18, 18] = 1.05
    p["Wm"][18, 10:18] = -1.5
    p["Wma"][18, 6] = -3.0
    p["bm"][18] = -11.8

    # Rotating exploration preference.
    angle = math.pi / 12.0
    radius = 1.02
    cs = radius * math.cos(angle)
    sn = radius * math.sin(angle)
    p["Wm"][19, 19] = cs
    p["Wm"][19, 20] = -sn
    p["Wm"][20, 19] = sn
    p["Wm"][20, 20] = cs
    p["bm"][19] = 0.3
    p["Wm"][19, 0] = -0.3

    # High gain converts tiny positive regrets into a bounded
    # rejection signal. No recurrent field receives this gain.
    def route(unit, candidate, orientation=0):
        for z_index in regret_units[candidate]:
            p["Wmz"][unit, z_index] = -1.0e35
        p["bm"][unit] = 1.0
        if orientation:
            p["Wm"][unit, 1] = 1000.0 * orientation
            p["bm"][unit] -= 1000.0

    route(21, 0)
    route(22, 1)
    route(23, 2, 1)
    route(24, 3, -1)
    route(25, 3, 1)
    route(26, 2, -1)

    # Blocker recovery controller.
    p["Wm"][27, 18] = 15.0
    p["Wmz"][27, 37] = 20.0
    p["Wmz"][27, 36] = -40.0
    p["bm"][27] = -28.0

    p["Wmz"][28, 36] = 20.0
    p["Wma"][28, 6] = -20.0
    p["bm"][28] = -10.0

    p["Wmz"][29, 36] = 20.0
    p["Wma"][29, 6] = 20.0
    p["bm"][29] = -30.0

    p["Wmz"][30, 36] = 20.0

    for unit, phase, sign in (
        (32, 20, 1.0),
        (33, 20, -1.0),
        (34, 19, -1.0),
        (35, 19, 1.0),
    ):
        p["Wm"][unit, phase] = 20.0 * sign
        p["Wm"][unit, 18] = 20.0
        p["bm"][unit] = -20.0

    # Action head.
    def output_mem(action, unit, weight):
        p["Wo"][action, 128 + unit] += weight

    output_mem(1, 21, 1.0)
    output_mem(2, 22, 1.0)

    output_mem(3, 23, 1.0)
    output_mem(3, 24, 1.0)
    p["bo"][3] = 1.0

    output_mem(4, 25, 1.0)
    output_mem(4, 26, 1.0)
    p["bo"][4] = 1.0

    for action in range(1, 5):
        output_mem(action, 30, -20.0)
        output_mem(action, 31 + action, 2.0)
        p["bo"][action] += 2.0

    # Move a selected blocker upward once, then deselect it.
    output_mem(1, 29, 20.0)
    p["bo"][1] += 20.0

    output_mem(6, 27, 20.0)
    output_mem(6, 28, 20.0)
    p["bo"][6] = 20.0

    for action in (0, 5, 7):
        p["bo"][action] = -1000000.0

    # Click a blocker normally, or a blue player to restore control.
    p["Wc"][0, 17] = 1.0
    p["Wc"][0, 16] = 2.0
    p["Wc"][0, 26] = -0.001

    return p
