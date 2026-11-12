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
    p = {k: np.zeros(s, np.float32) for k, s in shapes.items()}

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    index = {v: i for i, v in enumerate(directions)}
    M = 256.0
    pixel = 1.0 / 63.0
    shift = 3.5 / 63.0

    def c1(o, i, dy=0, dx=0, value=1.0):
        p["W1"][o, i, dy + 2, dx + 2] += value

    def c2(o, i, dy=0, dx=0, value=1.0):
        p["W2"][o, i, dy + 1, dx + 1] += value

    def rec(o, i, dy=0, dx=0, value=1.0):
        p["Wh"][o, i, dy + 1, dx + 1] += value

    # Local masks and coordinates.
    for color in (8, 9, 15):
        c1(0, color)
    c1(1, 9)
    c1(2, 1)
    for color in (11, 13):
        c1(3, color)
    c1(4, 12)
    c1(5, 6)
    c1(6, 17)
    c1(7, 16)

    for d, (dy, dx) in enumerate(directions):
        for color in (11, 13):
            c1(8 + d, color, 2 * dy, 2 * dx)

    def require_edge(o, coordinate, high):
        if high:
            c1(o, coordinate, value=64.0)
            p["b1"][o] -= 64.0
        else:
            c1(o, coordinate, value=-64.0)

    # Identify the progress stripe, including its two endpoints.
    for d in range(4):
        o = 12 + d
        tangent = (0, 1) if d < 2 else (1, 0)
        edge_coord = 16 if d < 2 else 17
        tangent_coord = 17 if d < 2 else 16
        high = d in (1, 3)

        for t in (-1, 0, 1):
            for color in (0, 14):
                c1(o, color, t * tangent[0], t * tangent[1])
        p["b1"][o] = -2.0
        require_edge(o, edge_coord, high)

        for endpoint in range(2):
            e = 20 + 2 * d + endpoint
            inward = 1 if endpoint == 0 else -1
            for t in (0, inward):
                for color in (0, 14):
                    c1(e, color, t * tangent[0], t * tangent[1])
            p["b1"][e] = -1.0
            require_edge(e, edge_coord, high)
            require_edge(e, tangent_coord, bool(endpoint))

    # Virtual lateral boundaries.
    c1(16, 17, value=-63.0)
    p["b1"][16] = 1.0
    c1(17, 17, value=63.0)
    p["b1"][17] = -62.0
    c1(18, 16, value=-63.0)
    p["b1"][18] = 1.0
    c1(19, 16, value=63.0)
    p["b1"][19] = -62.0

    # f2 masks, selected edges, endpoint coordinates, and cup mouths.
    for out, inp in enumerate((0, 2, 1, 5, 6, 7)):
        c2(out, inp)
    for inp in range(16, 20):
        c2(1, inp)

    for d, (dy, dx) in enumerate(directions):
        c2(6 + d, 1, -dy, -dx)
        c2(6 + d, 1, value=-1.0)

        coordinate = 6 if d < 2 else 7
        for complement in (0, 1):
            o = 10 + 2 * d + complement
            c2(o, 1, -dy, -dx, M)
            c2(o, 1, value=-M)
            c2(o, coordinate, value=-1.0 if complement else 1.0)
            p["b2"][o] = -M + float(complement)

        c2(18 + d, 2)
        c2(18 + d, 2, -dy, -dx, -1.0)

        def cup_three(out, vector, coefficient):
            j = index[vector]
            c2(out, 8 + j, vector[0], vector[1], coefficient)

        o = 22 + d
        perpendicular = (0, 1) if d < 2 else (1, 0)
        c2(o, 4)
        cup_three(o, (dy, dx), 1.0)
        cup_three(o, (-dy, -dx), -1.0)
        cup_three(o, perpendicular, 0.25)
        cup_three(o, (-perpendicular[0], -perpendicular[1]), 0.25)
        p["b2"][o] = -2.125

        c2(26 + d, 12 + d)
        c2(26 + d, 20 + 2 * d)
        c2(26 + d, 21 + 2 * d)

    def orientation(d):
        return 26 + d + (64 if d < 2 else 32)

    def feature(o, weights, bias=0.0):
        p["W3"][o] = weights
        p["b3"][o] = bias

    # h0:4: backward spill propagation.
    for d, (dy, dx) in enumerate(directions):
        v = np.zeros(96, np.float32)
        v[orientation(d)] = M
        v[0] -= M
        v[1] -= M
        v[22 + d] -= 16.0 * M
        v[18 + d] += M + 1.0
        feature(d, v)
        p["Uh"][d, d] = 1.0
        p["bh"][d] = -M
        rec(d, d, dy, dx)
        axis = 0 if d < 2 else 1
        rec(d, 4 + 2 * axis, dy, dx)
        rec(d, 5 + 2 * axis, dy, dx)

    # h4:8: lateral splitting at movable pieces.
    for axis in range(2):
        ds = (0, 1) if axis == 0 else (2, 3)
        perpendicular = (0, 1) if axis == 0 else (1, 0)
        for side_index, side in enumerate((-1, 1)):
            h = 4 + 2 * axis + side_index
            v = np.zeros(96, np.float32)
            v[0] = M
            for d in ds:
                v[orientation(d)] += M
            feature(h, v, -M)
            p["Uh"][h, h] = 1.0
            p["bh"][h] = -M

            dy = side * perpendicular[0]
            dx = side * perpendicular[1]
            rec(h, h, dy, dx)
            for d in ds:
                rec(h, d, dy, dx, 0.5)

    # Selected-piece endpoint geometry and source offsets.
    geometry = []
    for axis in range(2):
        ds = (0, 1) if axis == 0 else (2, 3)
        projection = 32 if axis == 0 else 64
        base = 8 + 7 * axis

        band = np.zeros(96, np.float32)
        maximum = np.zeros(96, np.float32)
        complement = np.zeros(96, np.float32)
        orient = np.zeros(96, np.float32)
        for d in ds:
            band[projection + 6 + d] += 1.0
            maximum[projection + 10 + 2 * d] += 1.0
            complement[projection + 11 + 2 * d] += 1.0
            orient[orientation(d)] += 1.0

        coordinate = np.zeros(96, np.float32)
        coordinate[4 + axis] = 1.0
        left = coordinate + complement
        right = coordinate - maximum
        left_bias = -1.0 + pixel
        right_bias = -pixel
        gate = M * (band + orient)

        feature(base, band + orient, -1.0)
        feature(base + 1, left + gate, left_bias + 2.0 - 2.0 * M)
        feature(base + 2, right + gate, right_bias + 2.0 - 2.0 * M)
        feature(base + 3, left + gate, left_bias + shift - 2.0 * M)
        feature(base + 4, left + gate, left_bias - shift - 2.0 * M)
        feature(base + 5, right + gate, right_bias + shift - 2.0 * M)
        feature(base + 6, right + gate, right_bias - shift - 2.0 * M)
        geometry.append(base)

        source_projection = 64 + 3 if axis == 0 else 32 + 3
        source_gate = gate.copy()
        source_gate[source_projection] += M
        feature(
            24 + 2 * axis, -left + source_gate,
            2.0 - left_bias - 3.0 * M
        )
        feature(
            25 + 2 * axis, right + source_gate,
            2.0 + right_bias - 3.0 * M
        )

    p["W3"][22, 2] = 1.0
    p["W3"][23, 3] = 1.0

    # Downstream depth on movable pieces.
    for d in range(4):
        v = np.zeros(96, np.float32)
        v[0] = M
        v[orientation(d)] = M
        coordinate = 5 if d < 2 else 4
        if d in (0, 2):
            v[coordinate] = 1.0
            bias = -2.0 * M
        else:
            v[coordinate] = -1.0
            bias = 1.0 - 2.0 * M
        feature(28 + d, v, bias)

    # h8:12: distances to safe lanes after either transverse shift.
    for i in range(4):
        endpoint = 0 if i < 2 else 1
        positive = bool(i % 2)
        s = shift if positive else -shift
        h = 8 + i
        for base in geometry:
            p["Uh"][h, base] += M
            p["Uh"][h, base + 1 + endpoint] += 1.0
            positive_part = base + (3 if endpoint == 0 else 5)
            p["Uh"][h, positive_part + int(positive)] -= 2.0
        p["bh"][h] = -s - M
        for d in range(4):
            rec(h, d, value=-10000.0)

    # Source and selected-piece spill readouts.
    p["Uh"][12, 23] = M
    p["bh"][12] = -M
    for d in range(4):
        rec(12, d)

    for axis in range(2):
        h = 13 + axis
        p["Uh"][h, 22] = M
        p["bh"][h] = -M
        rec(h, 4 + 2 * axis)
        rec(h, 5 + 2 * axis)

    for d in range(4):
        p["Uh"][15, 28 + d] = 1.0
    p["Uh"][15, 22] = 2.0
    for h in range(4, 8):
        rec(15, h, value=-1.0)

    nz = 0

    def zunit(weights=None, bias=0.0):
        nonlocal nz
        u = nz
        nz += 1
        if weights:
            for j, value in weights.items():
                p["Wz"][u, j] += value
        p["bz"][u] = bias
        return u

    def output(action, unit, coefficient):
        p["Wo"][action, unit] += coefficient

    difference = {9: 1.0, 11: 1.0, 8: -1.0, 10: -1.0}
    for axis, actions in enumerate(((3, 4), (1, 2))):
        for sign, action in zip((-1.0, 1.0), actions):
            w = {13 + axis: 1000.0}
            for j, value in difference.items():
                w[j] = sign * 100.0 * value
            output(action, zunit(w, -32.0), 1.0)

    depth_weights = {15: 1.0, 13: 1.0, 14: 1.0}
    depth = zunit(depth_weights, -2.0)
    deep = zunit(depth_weights, -2.54)

    # Source attraction is a second-stage, depth-gated computation.
    # This prevents the same hint from pulling a downstream outlet away
    # from its cups.
    source_window = 4.0 / 63.0
    source_baseline = math.tanh(-3.0)
    for axis, actions in enumerate(((3, 4), (1, 2))):
        for side in range(2):
            sf = 32 + 24 + 2 * axis + side
            terms = []
            for j, coefficient in enumerate((1.0, -2.0, 1.0)):
                terms.append((
                    zunit({sf: 1.0}, -2.0 - j * source_window),
                    coefficient
                ))

            m = 24 + 2 * axis + side
            p["bm"][m] = -3.0
            p["Wmz"][m, deep] = -10000.0
            for u, coefficient in terms:
                p["Wmz"][m, u] = 100.0 * coefficient

            for action, strength in (
                (actions[side], 30.0),
                (actions[1 - side], -15.0),
            ):
                p["Wo"][action, 128 + m] += strength
                p["bo"][action] -= strength * source_baseline

    safe = zunit({12: -10000.0}, 1.0)
    output(5, safe, 30.0)

    bad_units = (
        zunit({13: 100.0, 14: 100.0}),
        zunit({13: 100.0, 14: 100.0}, -1.0),
    )
    gravity_units = [
        zunit({32 + d: 1.0 / M}, -1.0 / M)
        for d in range(4)
    ]

    p["bo"][0] = -100.0
    p["bo"][7] = -100.0
    p["bo"][6] = 2.0

    # Prioritize costly downstream pieces; prefer upstream pieces once safe.
    for h in range(4, 8):
        p["Wc"][0, 32 + h] = 100.0
    for d in range(4):
        p["Wc"][0, 28 + d] = 2.0
    p["Wc"][0, 32 + 15] = -4.0
    p["Wc"][0, 22] = -0.1

    # Initial downstream-first selection.
    p["bm"][60] = 1.0
    p["Wm"][60, 60] = 5.0
    p["Wo"][6, 128 + 60] = -10000.0
    p["bo"][6] += 10000.0

    memory_one = math.tanh(3.0)
    for action in range(8):
        p["Wma"][action, action] = 3.0
    for action in (1, 2, 3, 4):
        p["Wo"][action, 128 + action] = 0.7 / memory_one

    # Reversals with residual spill try changing the piece ordering.
    pairs = ((1, 2), (2, 1), (3, 4), (4, 3))
    upstream = (1, 2, 3, 4)
    downstream = (2, 1, 4, 3)
    for d in range(4):
        for j, (current, previous) in enumerate(pairs):
            m = 8 + 4 * d + j
            p["Wma"][m, current] = 10.0
            p["Wm"][m, previous] = 10.0 / memory_one
            p["Wmz"][m, gravity_units[d]] = 10.0
            p["Wmz"][m, bad_units[0]] = 10.0
            p["Wmz"][m, bad_units[1]] = -10.0
            p["bm"][m] = -35.0
            p["Wo"][upstream[d], 128 + m] += 800.0
            p["bo"][upstream[d]] += 800.0

    # Transverse coordinate-change checks.
    position_weights = {
        32 + geometry[0] + 1: -1.0,
        32 + geometry[1] + 1: -1.0,
    }
    position = zunit(position_weights, 4.0)
    axis_units = [
        zunit({32 + geometry[axis]: 1.0})
        for axis in range(2)
    ]
    scale = 0.03
    p["Wmz"][36, position] = scale

    probes = (
        (3, 0, 1.0), (4, 0, -1.0),
        (1, 1, 1.0), (2, 1, -1.0),
    )
    for j, (action, axis, sign) in enumerate(probes):
        m = 37 + j
        gain = 5000.0
        p["Wmz"][m, position] = sign * gain
        p["Wm"][m, 36] = -sign * gain / scale
        p["Wmz"][m, axis_units[axis]] = 10000.0
        p["Wma"][m, action] = 10000.0
        p["bm"][m] = -20000.0 + gain * 0.018
        p["Wo"][6, 128 + m] += 1200.0
        p["bo"][6] += 1200.0

    # The same check for motion along gravity, including failed lifts.
    p["Wmz"][42, depth] = scale
    for d in range(4):
        for side, action in enumerate((upstream[d], downstream[d])):
            m = 43 + 2 * d + side
            sign = 1.0 if side == 0 else -1.0
            gain = 5000.0
            p["Wmz"][m, depth] = sign * gain
            p["Wm"][m, 42] = -sign * gain / scale
            p["Wmz"][m, gravity_units[d]] = 10000.0
            p["Wma"][m, action] = 10000.0
            p["bm"][m] = -20000.0 + gain * 0.018
            p["Wo"][6, 128 + m] += 1600.0
            p["bo"][6] += 1600.0

    # Do not immediately repeat a failed release.
    p["Wm"][61, 61] = 0.8
    p["Wma"][61, 5] = 3.0
    p["Wo"][5, 128 + 61] = -60.0

    return p
