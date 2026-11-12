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
    p = {k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()}
    colours = (8, 9, 12, 14)

    def a1(o, c, dy=0, dx=0, v=1.0):
        p["W1"][o, c, dy + 2, dx + 2] += v

    def a2(o, c, dy=0, dx=0, v=1.0):
        p["W2"][o, c, dy + 1, dx + 1] += v

    def a3(o, c, v=1.0):
        p["W3"][o, c] += v

    # Object coordinates: lower-right pixel of the central 2x2.
    perimeter = [
        (dy, dx)
        for dy in (-2, -1, 0, 1)
        for dx in (-2, -1, 0, 1)
        if not (dy in (-1, 0) and dx in (-1, 0))
    ]
    for j, colour in enumerate(colours):
        for dy, dx in perimeter:
            a1(j, colour, dy, dx)
            a1(4 + j, colour, dy, dx)
        p["b1"][j] = -11
        a1(4 + j, colour, -1, -1)
        p["b1"][4 + j] = -12

    # Selected and unselected segment-end motifs.
    motifs = {
        8: [(-1, -2, 2), (0, -2, 1)],
        9: [(-1, 2, 1), (0, 2, 2)],
        10: [(-2, -1, 1), (-2, 0, 2)],
        11: [(2, -1, 2), (2, 0, 1)],
        12: [(-1, -2, 3), (0, -2, 2)],
        13: [(-1, 2, 2), (0, 2, 3)],
        14: [(-2, -1, 2), (-2, 0, 3)],
        15: [(2, -1, 3), (2, 0, 2)],
    }
    for o, entries in motifs.items():
        for dy, dx, colour in entries:
            a1(o, colour, dy, dx)
        p["b1"][o] = -1

    corners = ((-2, -2), (-2, 1), (1, -2), (1, 1))
    core = ((-1, -1), (-1, 0), (0, -1), (0, 0))
    for o in (16, 17, 18):
        for dy, dx in corners:
            a1(o, 4 if o == 18 else 0, dy, dx)
        for dy, dx in core:
            for colour in (6, 15):
                a1(o, colour, dy, dx)
        p["b1"][o] = -7

    for colour in (6, 15):
        a1(16, colour, 0, 2, -1)
        a1(17, colour, 2, 0, -1)

    a1(19, 16)
    a1(20, 17)
    a1(21, 16, v=63)
    a1(22, 16, v=63)
    p["b1"][21] = -52
    p["b1"][22] = -53
    a1(23, 1)

    # A selected end immediately followed by a black obstacle/boundary.
    a1(24, 1)
    a1(24, 2, dy=1)
    for dy in (-2, -1, 0, 1, 2):
        for dx in (1, 2):
            a1(24, 5, dy, dx)
    p["b1"][24] = -11

    a1(25, 1)
    a1(25, 2, dx=-1)
    for dy in (1, 2):
        for dx in (-2, -1, 0, 1, 2):
            a1(25, 5, dy, dx)
    p["b1"][25] = -11

    def board(o):
        a2(o, 21, v=-1)
        a2(o, 22)

    def panel(o):
        p["b2"][o] -= 1
        a2(o, 21)
        a2(o, 22, v=-1)

    def horizontal(o, v=1):
        a2(o, 8, dx=-1, v=v)
        a2(o, 9, v=v)

    def vertical(o, v=1):
        a2(o, 10, dy=-1, v=v)
        a2(o, 11, v=v)

    for j in range(4):
        # Remaining beads exclude beads on the selected bar.
        a2(j, j)
        horizontal(j, -1)
        vertical(j, -1)
        board(j)

        a2(4 + j, j)
        horizontal(4 + j)
        p["b2"][4 + j] = -2
        board(4 + j)

        a2(8 + j, j)
        vertical(8 + j)
        p["b2"][8 + j] = -2
        board(8 + j)

        a2(12 + j, 4 + j)
        horizontal(12 + j)
        p["b2"][12 + j] = -2
        panel(12 + j)

    a2(16, 16)
    board(16)
    a2(17, 17)
    board(17)

    for j in range(4):
        a2(18, j)
        a2(18, 4 + j, v=-1)
    horizontal(18)
    p["b2"][18] = -2
    panel(18)

    a2(19, 23)
    board(19)

    # Carried beads pinned to a perpendicular unselected segment.
    for o in (20, 21):
        for j in range(4):
            a2(o, j)
        p["b2"][o] = -4
        board(o)

    horizontal(20)
    a2(20, 14, dy=-1)
    a2(20, 15)
    vertical(21)
    a2(21, 12, dx=-1)
    a2(21, 13)

    a2(22, 18)
    board(22)
    a2(23, 19)
    a2(24, 20)
    a2(25, 24)
    a2(25, 25)
    board(25)

    for j in range(4):
        # First unmatched goal: larger value means farther left.
        a3(j, 12 + j)
        a3(j, 24, -1)

        # Coordinate extrema of remaining beads.
        a3(4 + j, j)
        a3(4 + j, 23, -1)
        a3(8 + j, j)
        a3(8 + j, 23)
        p["b3"][8 + j] = -1

        a3(12 + j, j)
        a3(12 + j, 24, -1)
        a3(16 + j, j)
        a3(16 + j, 24)
        p["b3"][16 + j] = -1

        # Total carried-bead map.
        a3(20, 4 + j)
        a3(20, 8 + j)

        # Beads within the selected bar's perpendicular projection.
        a3(21, j)
        a3(22, j)

    # W3 input sections: local, rowmax, colmax.
    a3(21, 64 + 19)
    a3(22, 32 + 19)
    p["b3"][21] = -1
    p["b3"][22] = -1
    a3(23, 25)

    a3(24, 16)
    a3(24, 23)
    p["b3"][24] = -1
    a3(25, 17)
    a3(25, 24)
    p["b3"][25] = -1
    a3(26, 16)
    a3(27, 17)
    a3(28, 19)
    a3(29, 18)
    a3(30, 20)
    a3(30, 21)
    a3(31, 22)
    p["Wc"][0, 31] = 1

    # With Wh=0, these channels are stable local conjunctions.
    for j in range(4):
        for out, projection in ((j, 21), (4 + j, 22)):
            p["Uh"][out, projection] = 1
            p["Uh"][out, 4 + j] = 1
            p["Uh"][out, 8 + j] = 1
            p["bh"][out] = -1

    def expression(coeff=None, bias=0):
        return ({} if coeff is None else dict(coeff), float(bias))

    def add(*terms):
        coeff, bias = {}, 0.0
        for a, b in terms:
            bias += b
            for k, v in a.items():
                coeff[k] = coeff.get(k, 0.0) + v
        return coeff, bias

    def mul(term, factor):
        a, b = term
        return {k: v * factor for k, v in a.items()}, b * factor

    def constant(value):
        return expression(bias=value)

    def maximum(c):
        return expression({32 + c: 1})

    def hidden_maximum(c):
        return expression({c: 1})

    def count(c):
        return expression({64 + c: 4096})

    nz = 0

    def feature(term):
        nonlocal nz
        o = nz
        nz += 1
        if nz > 128:
            raise RuntimeError("Too many features")
        a, b = term
        for k, v in a.items():
            p["Wz"][o, k] = v
        p["bz"][o] = b
        return o

    goals = [maximum(j) for j in range(4)]
    goal_units = [feature(g) for g in goals]
    comparisons = [[] for _ in range(4)]

    for j in range(4):
        for k in range(4):
            if j != k:
                comparisons[j].append(feature(add(
                    goals[k], mul(goals[j], -1), constant(0.02)
                )))

    head_h, head_v = maximum(24), maximum(25)
    flag_h = feature(maximum(26))
    flag_v = feature(maximum(27))
    beads = count(20)
    matches = count(29)

    excess = feature(add(beads, mul(matches, -1), constant(-0.5)))
    length = add(
        mul(count(28), 0.125),
        mul(beads, 0.75),
        constant(0.875),
    )
    packed = feature(add(
        matches, constant(1.3), mul(length, -1)
    ))
    pinned = feature(count(30))
    match_count = feature(matches)
    bead_count = feature(beads)
    blocked = feature(maximum(23))

    projected = {}
    for j in range(4):
        projected["H", j] = feature(hidden_maximum(j))
        projected["V", j] = feature(hidden_maximum(4 + j))

    missing = [
        feature(add(
            constant(0.75),
            mul(maximum(4 + j), -1),
            mul(maximum(8 + j), -1)
        ))
        for j in range(4)
    ]

    distance, direction, nearest = {}, {}, {}
    for orientation in ("H", "V"):
        head = head_h if orientation == "H" else head_v
        lo_base = 4 if orientation == "H" else 12
        hi_base = 8 if orientation == "H" else 16

        for j in range(4):
            low = add(constant(1), mul(maximum(lo_base + j), -1))
            high = maximum(hi_base + j)
            twice_midpoint = add(low, high)

            nearest[orientation, j, 0] = feature(add(
                mul(head, 2), mul(twice_midpoint, -1),
                constant(0.001)
            ))
            nearest[orientation, j, 1] = feature(add(
                twice_midpoint, mul(head, -2),
                constant(0.001)
            ))

            for side, target in enumerate((low, high)):
                delta = add(target, mul(head, -1))
                distance[orientation, j, side] = (
                    feature(delta), feature(mul(delta, -1))
                )
                direction[orientation, j, side] = (
                    feature(add(delta, constant(0.025))),
                    feature(add(mul(delta, -1), constant(0.025)))
                )

    p["bo"][:] = [-100, 0, 0, 0, 0, -100, -0.1, -100]
    nm = 0

    def decision(coeff, bias, action, vote=1, sharpness=60,
                 recurrent=None):
        nonlocal nm
        o = nm
        nm += 1
        if nm > 60:
            raise RuntimeError("Too many decisions")
        for k, v in coeff.items():
            p["Wmz"][o, k] = sharpness * v
        if recurrent:
            for k, v in recurrent.items():
                p["Wm"][o, k] = sharpness * v
        p["bm"][o] = sharpness * bias
        p["Wo"][action, 128 + o] += vote
        p["bo"][action] += vote
        return o

    def put(coeff, unit, value):
        coeff[unit] = coeff.get(unit, 0.0) + value

    perpendicular_units = []

    for orientation in ("H", "V"):
        flag = flag_h if orientation == "H" else flag_v
        actions = (1, 2, 4) if orientation == "H" else (3, 4, 2)

        for j in range(4):
            for side in range(2):
                for kind, action in enumerate(actions):
                    coeff = {flag: 5000}
                    for unit in comparisons[j]:
                        put(coeff, unit, -100000)
                    for unit in (excess, pinned, missing[j]):
                        put(coeff, unit, -5000)
                    put(coeff, nearest[orientation, j, side], -10000)

                    if kind < 2:
                        put(
                            coeff,
                            direction[orientation, j, side][kind],
                            -1000
                        )
                        put(coeff, projected[orientation, j], 80)
                        unit = decision(
                            coeff, -4999.90, action,
                            recurrent={60: -40, 62: -40}
                        )
                        perpendicular_units.append(unit)
                    else:
                        for unit in distance[orientation, j, side]:
                            put(coeff, unit, -1000)
                        put(coeff, projected[orientation, j], -3000)
                        decision(
                            coeff, -4999.90, action,
                            recurrent={60: 1000, 62: 1000}
                        )

    # Memory 60: detour latch.
    # Memory 61: previous match count, encoded at small amplitude.
    # Memory 62: initialization/constant-one flag.
    # Memory 63: compaction latch.
    p["Wmz"][61, match_count] = 0.02
    p["bm"][62] = 10

    p["Wm"][60, 60] = 10
    p["Wm"][60, 63] = -15
    p["Wm"][60, 62] = -15
    p["Wmz"][60, blocked] = 30
    p["Wmz"][60, bead_count] = -100
    p["Wma"][60, 6] = -100
    p["bm"][60] = -5

    phase_coeff = {
        match_count: 2,
        excess: 4,
        packed: -100,
    }
    phase_rec = {
        61: -100,
        62: len(perpendicular_units) + 0.5,
        63: 0.5,
    }
    for unit in perpendicular_units:
        phase_rec[unit] = 1

    for k, v in phase_coeff.items():
        p["Wmz"][63, k] = 20 * v
    for k, v in phase_rec.items():
        p["Wm"][63, k] = 20 * v
    p["bm"][63] = -10

    for flag, action in ((flag_h, 3), (flag_v, 1)):
        coeff = dict(phase_coeff)
        put(coeff, flag, 200)
        put(coeff, pinned, -200)
        decision(
            coeff, -200.5, action,
            vote=5, sharpness=30, recurrent=phase_rec
        )

    # Clear a newly caught wrong bead before retracting.
    for flag, action in ((flag_h, 2), (flag_v, 4)):
        decision(
            {flag: 200, excess: 30},
            -200.5, action, vote=10, sharpness=30,
            recurrent={62: -100, 63: -100}
        )

    # Release a bead held by a perpendicular crossing.
    for flag, action in ((flag_h, 2), (flag_v, 4)):
        decision(
            {flag: 200, pinned: 30},
            -201, action, vote=7, sharpness=30
        )

    decision(
        {unit: -1 for unit in goal_units},
        0.035, 6, vote=5, sharpness=150
    )

    decision({flag_h: 1}, -0.5, 4, vote=0.025)
    decision({flag_v: 1}, -0.5, 2, vote=0.025)

    # Start a negative-perpendicular detour around a blocked tip.
    for flag, action in ((flag_h, 1), (flag_v, 3)):
        decision(
            {flag: 200, blocked: 30, bead_count: -100},
            -200.5, action, vote=10, sharpness=30
        )

    return p
