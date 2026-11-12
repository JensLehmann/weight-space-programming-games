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

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    opposite = [1, 0, 3, 2]

    def w1(out, inp, dy=0, dx=0, value=1.0):
        p["W1"][out, inp, dy + 2, dx + 2] += value

    def w2(out, inp, dy=0, dx=0, value=1.0):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    def wh(out, inp, dy=0, dx=0, value=1.0):
        p["Wh"][out, inp, dy + 1, dx + 1] += value

    # Road, exit, enemy bodies, and integer coordinate codes.
    for color in (0, 2, 4, 8, 9, 11, 12, 13, 14, 15):
        w1(0, color)
    w1(1, 14)
    for color in (8, 12, 13):
        w1(2, color)
    w1(3, 16, value=63.0)
    w1(4, 17, value=63.0)
    p["b1"][3:5] = 128.0

    for d, (dy, dx) in enumerate(directions):
        code = 5 + d
        legal = 17 + d

        # Complete player tile and an outgoing connector.
        for ry in range(-1, 2):
            for rx in range(-1, 2):
                for color in (9, 4):
                    w1(code, color, ry, rx, 256.0)
                    w1(legal, color, ry, rx)
        w1(code, 2, 2 * dy, 2 * dx, 256.0)
        w1(legal, 2, 2 * dy, 2 * dx)
        w1(code, 16 if dy else 17, value=63.0)
        p["b1"][code] = -2432.0
        p["b1"][legal] = -9.0

        # All immediate threats: chargers, patrols, active followers.
        all_normal = 9 + 2 * d
        all_follower = all_normal + 1

        # Static route obstacles exclude patrols.
        static_normal = 24 + 2 * d
        static_follower = static_normal + 1

        for ry in range(-1, 2):
            for rx in range(-1, 2):
                if (ry, rx) == (dy, dx):
                    continue
                oy, ox = ry - dy, rx - dx
                w1(all_normal, 8, oy, ox)
                w1(all_normal, 12, oy, ox)
                w1(all_follower, 13, oy, ox)
                w1(static_normal, 8, oy, ox)
                w1(static_follower, 13, oy, ox)

        for color in (15, 11):
            w1(all_normal, color)
            w1(static_normal, color)
        w1(all_follower, 11)
        w1(static_follower, 11)

        for channel in (all_normal, all_follower,
                        static_normal, static_follower):
            p["b1"][channel] = -8.0

    # Full patrol-tile detections carrying row, column, and presence.
    for ry in range(-1, 2):
        for rx in range(-1, 2):
            for color in (12, 15, 11):
                w1(21, color, ry, rx, 256.0)
                w1(22, color, ry, rx, 256.0)
                w1(23, color, ry, rx)
    w1(21, 16, value=63.0)
    w1(22, 17, value=63.0)
    p["b1"][21:23] = -2176.0
    p["b1"][23] = -8.0

    # Second-layer primitives.
    for c in range(5):
        w2(c, c)

    for d, (dy, dx) in enumerate(directions):
        py, px = -dx, dy
        for offset in (-1, 0, 1):
            w2(5 + d, 5 + d, offset * py, offset * px)
            w2(13 + d, 17 + d, offset * py, offset * px)

            sy = -dy + offset * py
            sx = -dx + offset * px
            w2(9 + d, 9 + 2 * d, sy, sx)
            w2(9 + d, 10 + 2 * d, sy, sx)
            w2(20 + d, 24 + 2 * d, sy, sx)
            w2(20 + d, 25 + 2 * d, sy, sx)

    for ry in range(-1, 2):
        for rx in range(-1, 2):
            w2(17, 21, ry, rx)
            w2(18, 22, ry, rx)
            w2(19, 23, ry, rx)

    # Third-layer road and exit features.
    p["W3"][0, 0] = 1.0
    p["W3"][1, 1] = 1.0

    for d, (dy, dx) in enumerate(directions):
        p["W3"][3 + d, 9 + d] = 1.0
        p["W3"][19 + d, 5 + d] = 1.0
        p["W3"][27 + d, 20 + d] = 1.0

        # Axial transport of legal-player coordinate codes.
        axial = (64 if dy else 32) + 5 + d
        coordinate = 3 if dy else 4
        displacement = 6 * (dy if dy else dx)

        for k, offset in enumerate((2.0, 0.0, -2.0)):
            out = 7 + 3 * d + k
            p["W3"][out, axial] = 1.0
            p["W3"][out, coordinate] = -1.0
            p["b3"][out] = displacement + offset

        # At a legal player position, encode 128 + signed distance
        # to an axially aligned patrol. Elsewhere this is zero.
        out = 23 + d
        axis_offset = 64 if dy else 32
        patrol_coordinate = 17 if dy else 18
        sign = dy if dy else dx
        p["W3"][out, 13 + d] = 512.0
        p["W3"][out, axis_offset + 19] = 512.0
        p["W3"][out, axis_offset + patrol_coordinate] = sign
        p["W3"][out, 5 + d] = -sign
        p["b3"][out] = -896.0

    # Recurrent directional danger projections. Each starts just
    # beyond the enemy's current node and covers its next node.
    static_h = [6, 9, 10, 15]
    for d, (dy, dx) in enumerate(directions):
        wh(d, d, -dy, -dx)
        p["Uh"][d, 3 + d] = 7.0
        p["bh"][d] = -1.0

        out = static_h[d]
        wh(out, out, -dy, -dx)
        p["Uh"][out, 27 + d] = 7.0
        p["bh"][out] = -1.0

    # Clock and delayed exit seed.
    wh(4, 4)
    p["bh"][4] = 1.0
    wh(5, 4)
    p["Uh"][5, 1] = 256.0
    p["bh"][5] = -264.0

    # Bounded exit reachability = raw minus overflow.
    for out in (7, 8):
        for ry in range(-1, 2):
            for rx in range(-1, 2):
                wh(out, 7, ry, rx)
                wh(out, 8, ry, rx, -1.0)
        wh(out, 5)
        for hazard in static_h:
            wh(out, hazard, value=-256.0)
        p["Uh"][out, 0] = 10.0
        p["bh"][out] = -10.0
    p["bh"][8] -= 1.0

    # Destination accumulators: safe baseline plus exit arrival age.
    # Immediate collisions erase the entire destination score.
    for d in range(4):
        out = 11 + d
        wh(out, out)
        wh(out, 7)
        wh(out, 8, value=-1.0)
        for hazard in range(4):
            wh(out, hazard, value=-4.0)

        ramp = 7 + 3 * d
        p["Uh"][out, ramp] = 2.0
        p["Uh"][out, ramp + 1] = -4.0
        p["Uh"][out, ramp + 2] = 2.0
        p["bh"][out] = -3.0

    for d in range(4):
        action = 1 + d
        p["Wz"][d, 11 + d] = 1.0
        p["Wo"][action, d] = 1.0

        # Small capped bonus for legal moves.
        feature = 32 + 19 + d
        a, b = 4 + 2 * d, 5 + 2 * d
        p["Wz"][a, feature] = 1.0
        p["Wz"][b, feature] = 1.0
        p["bz"][b] = -1.0
        p["Wo"][action, a] = 0.001
        p["Wo"][action, b] = -0.001

        # Triangular approach penalty, supported at directed patrol
        # distances 0..24 pixels and peaking at 12 pixels.
        feature = 32 + 23 + d
        for k, threshold in enumerate((128.0, 140.0, 152.0)):
            unit = 16 + 3 * d + k
            p["Wz"][unit, feature] = 1.0
            p["bz"][unit] = -threshold
        p["Wo"][action, 16 + 3 * d] = -1.3
        p["Wo"][action, 17 + 3 * d] = 2.6
        p["Wo"][action, 18 + 3 * d] = -1.3

    # Reset only if there is no detected legal outgoing move.
    p["bz"][12] = 1.0
    for d in range(4):
        p["Wz"][12, 32 + 19 + d] = -1.0 / 128.0
    p["Wo"][0, 12] = 1.0
    p["bo"][0] = -0.5
    p["bo"][5:] = -1_000_000.0

    # Memory 0..3 stores the last action as +/-1 indicators.
    for d in range(4):
        p["Wma"][d, 1 + d] = 20.0
        p["bm"][d] = -10.0

    # Memory 4..19 recognizes alternating histories of lengths 2..5.
    for length in range(2, 6):
        base = 4 + 4 * (length - 2)
        previous_base = 0 if length == 2 else base - 4
        for d in range(4):
            unit = base + d
            p["Wma"][unit, 1 + d] = 10.0
            p["Wm"][unit, previous_base + opposite[d]] = 5.0
            p["bm"][unit] = -12.0

    # A decaying escape signal is triggered by a long alternating run.
    # The validity bit suppresses a spurious episode-start trigger.
    p["Wm"][20, 20] = 0.95
    p["Wm"][20, 16:20] = 2.0
    p["Wm"][20, 21] = 8.0
    p["bm"][21] = 10.0

    for d in range(4):
        action = 1 + d

        # Discourage completing the detected alternating sequence.
        p["Wo"][action, 128 + 16 + opposite[d]] = -6.0

        # While escaping, discourage reversing the latest movement.
        unit = 22 + d
        p["Wma"][unit, 1 + opposite[d]] = 10.0
        p["Wm"][unit, 20] = 20.0
        p["bm"][unit] = -15.0
        p["Wo"][action, 128 + unit] = -7.5

    # Subordinate deterministic heading preferences.
    for previous, (py, px) in enumerate(directions):
        right = (px, -py)
        left = (-px, py)
        for d, direction in enumerate(directions):
            if direction == right:
                preference = 0.0003
            elif direction == (py, px):
                preference = 0.0002
            elif direction == left:
                preference = 0.0001
            else:
                preference = 0.0
            p["Wo"][1 + d, 128 + previous] += preference / 2.0

    p["bo"][1] = 0.00005
    p["bo"][4] = 0.00002
    return p
