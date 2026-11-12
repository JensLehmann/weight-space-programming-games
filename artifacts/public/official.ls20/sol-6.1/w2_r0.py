import math
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
    p = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in shapes.items()
    }

    colours = (12, 9, 14, 8)

    # Rotation-invariant linear shape separators:
    # corner count, edge count, centre bit, threshold.
    tests = [
        (1, -1, 4, 4),
        (-1, -2, 10, 3),
        (1, 1, -4, 6),
        (-1, 1, -4, 2),
        (-1, 1, 4, 5),
        (1, -1, -4, 1),
    ]

    # First convolution.
    GSH = list(range(6))
    CSH = list(range(6, 12))
    CCODE, CMY, CMX = 12, 13, 14
    GCODE, GMY, GMX = 15, 16, 17
    PLAYER, RAW_CHARGE = 18, 19
    SHAPER, COLOURER, ROTATOR = 20, 21, 22
    WALL_RAW, WALL_EXCESS = 23, 24
    DISPLAY_COORDS = (25, 26, 27, 28)
    UI_EXCLUSION = 29
    ENERGY_COORDS = (30, 31)

    def w1(out, colour, dy, dx, value):
        p["W1"][out, colour, dy + 2, dx + 2] += value

    def pixel_kind(dy, dx):
        if dy == 0 and dx == 0:
            return 2
        return 0 if dy and dx else 1

    for s, (cw, ew, qw, threshold) in enumerate(tests):
        weights = (cw, ew, qw)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                value = weights[pixel_kind(dy, dx)]
                for colour in colours:
                    w1(GSH[s], colour, dy, dx, value)
                    w1(CSH[s], colour, 2 * dy, 2 * dx, value)

        # Goals have a dark border around the 3x3 shape.
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if abs(dy) == 2 or abs(dx) == 2:
                    w1(GSH[s], 5, dy, dx, 20.0)

        p["b1"][GSH[s]] = -320.0 - threshold
        p["b1"][CSH[s]] = -threshold

    # Summed colour codes.
    for ci, colour in enumerate(colours, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                w1(CCODE, colour, 2 * dy, 2 * dx, ci)
                w1(GCODE, colour, dy, dx, ci)

    # Orientation-sensitive signed moments.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            factor = 1.0 if dy and dx else 2.0
            for colour in colours:
                w1(CMY, colour, 2 * dy, 2 * dx, factor * dy)
                w1(CMX, colour, 2 * dy, 2 * dx, factor * dx)
                w1(GMY, colour, dy, dx, factor * dy)
                w1(GMX, colour, dy, dx, factor * dx)

    for out in (CMY, CMX, GMY, GMX):
        p["b1"][out] = 4.0

    # Exact player centre detector.
    for dy in range(-2, 3):
        colour = 12 if dy < 0 else 9
        for dx in range(-2, 3):
            w1(PLAYER, colour, dy, dx, 1.0)
    p["b1"][PLAYER] = -24.0

    w1(RAW_CHARGE, 11, 0, 0, 1.0)

    def exact_marker(out, required):
        for colour in set(required.values()):
            p["W1"][out, colour, :, :] = -2.0
        for (dy, dx), colour in required.items():
            p["W1"][out, colour, dy + 2, dx + 2] = 2.0
        p["b1"][out] = 1.0 - 2.0 * len(required)

    exact_marker(SHAPER, {
        (-1, -1): 0,
        (0, 0): 0,
        (0, 1): 0,
        (1, 0): 0,
    })
    exact_marker(ROTATOR, {
        (-1, 0): 0,
        (0, 0): 0,
        (0, 1): 0,
        (0, -1): 1,
        (1, 0): 1,
    })
    exact_marker(COLOURER, {
        (-1, -1): 9, (-1, 0): 14, (-1, 1): 14,
        (0, -1): 9, (0, 0): 0, (0, 1): 8,
        (1, -1): 12, (1, 0): 12, (1, 1): 8,
    })

    # Their difference is a binary 5x5 clearance mask.
    for out in (WALL_RAW, WALL_EXCESS):
        p["W1"][out, 4, :, :] = 1.0
    p["b1"][WALL_EXCESS] = -1.0

    # Coordinate hinges locating the fixed display.
    p["W1"][25, 16, 2, 2] = 63.0
    p["b1"][25] = -57.0
    p["W1"][26, 16, 2, 2] = -63.0
    p["b1"][26] = 57.0
    p["W1"][27, 17, 2, 2] = 63.0
    p["b1"][27] = -5.0
    p["W1"][28, 17, 2, 2] = -63.0
    p["b1"][28] = 5.0

    p["W1"][UI_EXCLUSION, 16, 2, 2] = 63.0
    p["b1"][UI_EXCLUSION] = -53.0

    p["W1"][30, 16, 2, 2] = 63.0
    p["b1"][30] = -61.0
    p["W1"][31, 16, 2, 2] = -63.0
    p["b1"][31] = 61.0

    # Second convolution / axial features.
    BLOCKED = (0, 1, 2, 3)  # up, down, left, right
    GOAL = 5
    GRANK, GC, GY, GX = 6, 7, 8, 9
    CS = list(range(10, 16))
    CC, CY, CX = 16, 17, 18
    PL, RE = 19, 20
    ST_SH, ST_CO, ST_RO = 21, 22, 23
    WA = 24
    PROBE_R, PROBE_L, PROBE_D, PROBE_U = 25, 26, 27, 28
    ENERGY = 29
    K = 64.0

    def w2(out, inp, dy=0, dx=0, value=1.0):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    for s in range(6):
        w2(GOAL, GSH[s])
        w2(GRANK, GSH[s], value=6.0 - s)
        for out in (GC, GY, GX):
            w2(out, GSH[s], value=K * (6 - s))

    w2(GC, GCODE)
    p["b2"][GC] = -32.0
    w2(GY, GMY)
    w2(GX, GMX)
    p["b2"][GY] = -20.0
    p["b2"][GX] = -20.0

    # Exclude the status display from goal recognition.
    for out in (GOAL, GRANK, GC, GY, GX):
        w2(out, UI_EXCLUSION, value=-10000.0)

    current_pairs = (
        [(CS[s], CSH[s]) for s in range(6)]
        + [(CC, CCODE), (CY, CMY), (CX, CMX)]
    )
    for out, inp in current_pairs:
        w2(out, inp)
        for coord in DISPLAY_COORDS:
            w2(out, coord, value=-10.0)

    for out, inp in (
        (PL, PLAYER),
        (ST_SH, SHAPER),
        (ST_CO, COLOURER),
        (ST_RO, ROTATOR),
    ):
        w2(out, inp)

    # Recharge ring.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            w2(
                RE, RAW_CHARGE, dy, dx,
                -8.0 if dy == 0 and dx == 0 else 1.0,
            )
    p["b2"][RE] = -7.0

    def wall_at(out, dy, dx):
        w2(out, WALL_RAW, dy, dx, 1.0)
        w2(out, WALL_EXCESS, dy, dx, -1.0)

    wall_at(WA, 0, 0)
    for out, (dy, dx) in zip(
        BLOCKED, ((-1, 0), (1, 0), (0, -1), (0, 1))
    ):
        wall_at(out, dy, dx)
        w2(out, PLAYER)
        p["b2"][out] = -1.0

    # Two probe locations on each axis.
    w2(PROBE_R, PLAYER, 0, -1)
    w2(PROBE_L, PLAYER, 0, 1)
    w2(PROBE_D, PLAYER, -1, 0)
    w2(PROBE_U, PLAYER, 1, 0)

    # The sum along row 61 is the remaining energy count.
    w2(ENERGY, RAW_CHARGE)
    for coord in ENERGY_COORDS:
        w2(ENERGY, coord, value=-10.0)

    for ch in range(30):
        p["W3"][ch, ch] = 1.0

    # Filter lower-priority aligned goals and their attributes.
    for ch in (GOAL, GC, GY, GX):
        p["W3"][ch, GRANK] += 2000.0
        p["W3"][ch, 32 + GRANK] -= 1000.0
        p["W3"][ch, 64 + GRANK] -= 1000.0

    # Four independent recurrent fields.
    source_sets = [
        [GOAL, RE],
        [ST_SH],
        [ST_CO],
        [ST_RO],
    ]
    neighbours = (
        (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)
    )

    for field, sources in enumerate(source_sets):
        a, b, hx, hy = (4 * field + j for j in range(4))

        # a-b is clipped reachability; the player does not interrupt it.
        for out in (a, b):
            for dy, dx in neighbours:
                p["Wh"][out, a, dy + 1, dx + 1] = 1.0
                p["Wh"][out, b, dy + 1, dx + 1] = -1.0
            for src in sources:
                p["Uh"][out, src] += 1.0
            p["Uh"][out, WA] = -10000.0
        p["bh"][b] = -1.0

        # Each probe accumulates its local reachability over time.
        # Different baselines let max and mean recover both sides.
        for out in (hx, hy):
            p["Wh"][out, out, 1, 1] = 1.0
            p["Wh"][out, a, 1, 1] = 1.0
            p["Wh"][out, b, 1, 1] = -1.0
            p["bh"][out] = -1024.0

        p["Uh"][hx, PROBE_R] = 1026.0
        p["Uh"][hx, PROBE_L] = 1025.0
        p["Uh"][hy, PROBE_D] = 1026.0
        p["Uh"][hy, PROBE_U] = 1025.0

    def feature(ch):
        return 32 + ch

    z_next = 0

    def neuron(coefficients, bias):
        nonlocal z_next
        out = z_next
        z_next += 1
        for index, value in coefficients.items():
            p["Wz"][out, index] += value
        p["bz"][out] = bias
        return out

    def navigation(field, gate, bias, strength):
        hx = 4 * field + 2
        hy = 4 * field + 3

        # Recover the four arrival counts from two multiplexed probes.
        specifications = (
            (1, {16 + hy: 4096.0, hy: -1.0}, -128.0),
            (2, {hy: 1.0}, -256.0),
            (3, {16 + hx: 4096.0, hx: -1.0}, -128.0),
            (4, {hx: 1.0}, -256.0),
        )
        for action, arrival, offset in specifications:
            coefficients = dict(gate)
            for index, value in arrival.items():
                coefficients[index] = coefficients.get(index, 0.0) + value
            z = neuron(coefficients, bias + offset)
            p["Wo"][action, z] += strength

    M = 1024.0

    # Shape mismatch selects the shape-changing field.
    difference = {feature(GRANK): 1.0}
    for s in range(6):
        difference[feature(CS[s])] = -(6.0 - s)

    for sign in (-1.0, 1.0):
        gate = {i: sign * M * v for i, v in difference.items()}
        navigation(1, gate, -0.25 * M, 512.0)

    # Colour comparison is exact once the shape counts agree.
    difference = {
        feature(GC): 1.0,
        feature(GRANK): -K,
        feature(CC): -1.0,
    }
    for sign in (-1.0, 1.0):
        gate = {i: sign * M * v for i, v in difference.items()}
        navigation(
            2, gate, sign * 32.0 * M - 0.25 * M, 32.0
        )

    # Two signed moments distinguish the four orientations.
    for goal_moment, current_moment in ((GY, CY), (GX, CX)):
        difference = {
            feature(goal_moment): 1.0,
            feature(GRANK): -K,
            feature(current_moment): -1.0,
        }
        for sign in (-1.0, 1.0):
            gate = {i: sign * M * v for i, v in difference.items()}
            navigation(
                3, gate, sign * 20.0 * M - 0.25 * M, 4.0
            )

    navigation(0, {}, 1024.0, 1.0)

    # Energy override, enabled only while recharge markers remain.
    energy_mean_index = 64 + ENERGY
    urgent_gate = {
        feature(RE): 65536.0,
        energy_mean_index: -4096.0 * M,
    }
    navigation(
        0,
        urgent_gate,
        8.75 * M - 65536.0,
        4096.0,
    )

    # Hard collision penalties.
    for action, blocked_channel in zip((1, 2, 3, 4), BLOCKED):
        z = neuron({feature(blocked_channel): 1.0}, 0.0)
        p["Wo"][action, z] = -1.0e8

    p["bo"][:] = -1.0e10
    p["bo"][1:5] = 0.0

    # Avoid immediate reversals when arrival times tie.
    for action in range(1, 5):
        p["Wma"][action, action] = 3.0
    reverse = {1: 2, 2: 1, 3: 4, 4: 3}
    for previous, action in reverse.items():
        p["Wo"][action, 128 + previous] = -0.2

    # Small deterministic tie-breaking oscillator.
    theta = math.pi / 24.0
    gain = 1.015
    p["Wm"][5, 5] = gain * math.cos(theta)
    p["Wm"][5, 6] = -gain * math.sin(theta)
    p["Wm"][6, 5] = gain * math.sin(theta)
    p["Wm"][6, 6] = gain * math.cos(theta)
    p["bm"][5] = 0.01
    p["Wo"][1, 134] = -0.02
    p["Wo"][2, 134] = 0.02
    p["Wo"][3, 133] = -0.02
    p["Wo"][4, 133] = 0.02

    return p
