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

    def c1(o, i, v=1.0, dr=0, dc=0):
        p["W1"][o, i, 2 + dr, 2 + dc] += v

    def c2(o, i, v=1.0, dr=0, dc=0):
        p["W2"][o, i, 1 + dr, 1 + dc] += v

    def f3(o, i, v=1.0, pool=0):
        p["W3"][o, i + 32 * pool] += v

    def rec(o, i, v=1.0, dr=0, dc=0):
        p["Wh"][o, i, 1 + dr, 1 + dc] += v

    # ---------------------------------------------------------
    # Local palette, foot edges, goal paint and gate endpoints.
    # ---------------------------------------------------------
    c1(0, 11)
    c1(0, 14)
    c1(1, 0)
    c1(2, 3)
    c1(3, 5)
    c1(4, 9)
    c1(5, 1)
    c1(5, 12)
    c1(6, 12)

    c1(7, 11, 8)
    c1(7, 14, 8)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        c1(7, 5, 1, dr, dc)
        c1(7, 4, -4, dr, dc)
    p["b1"][7] = -8

    # Direction toward the yellow body, opposite gravity.
    for o, dr, dc in ((8, -1, 0), (9, 0, 1), (10, 0, -1)):
        c1(o, 11, 8)
        c1(o, 14, 8)
        c1(o, 4, 1, 2 * dr, 2 * dc)
        c1(o, 0, 4, -dr, -dc)
        c1(o, 18, -4, -dr, -dc)
        p["b1"][o] = -8

    for o, dr, dc in ((11, 1, 0), (12, 0, -1)):
        c1(o, 1)
        c1(o, 12)
        c1(o, 5, 1, dr, dc)
        p["b1"][o] = -1

    c1(13, 16, -63)
    p["b1"][13] = 1

    for o, coordinate in ((14, 17), (15, 16)):
        c1(o, coordinate)
        c1(o, 11, 2)
        c1(o, 14, 2)
        p["b1"][o] = -2

    directions = ((0, 1), (0, -1), (1, 0), (-1, 0))
    for j, (dr, dc) in enumerate(directions):
        c1(16 + j, 9)
        c1(16 + j, 9, 1, dr, dc)
        c1(16 + j, 5, 1, 2 * dr, 2 * dc)
        p["b1"][16 + j] = -2

        c1(24 + j, 9)
        c1(24 + j, 5, 1, dr, dc)
        p["b1"][24 + j] = -1

    # Material sample on the upper side of a horizontal divider.
    c1(22, 0, 1, -2, 0)

    c1(28, 1)
    c1(28, 5, 1, 1, 0)
    p["b1"][28] = -1
    c1(29, 1)
    c1(29, 5, 1, 0, -1)
    p["b1"][29] = -1

    c1(30, 17)
    c1(31, 16)

    # ---------------------------------------------------------
    # Second local layer.
    # ---------------------------------------------------------
    c2(1, 4)
    c2(2, 6)
    c2(3, 7)

    c2(4, 14)
    c2(4, 8)
    p["b2"][4] = -1
    c2(28, 15)
    c2(28, 9)
    p["b2"][28] = -1

    c2(5, 8)
    c2(7, 9)
    c2(9, 10)
    c2(6, 30)
    c2(8, 31)

    # A clipped, expanded band around horizontal gates.
    for dr in (-1, 0, 1):
        c2(27, 12, 1, dr, 0)
        c2(10, 12, 1, dr, 0)
    p["b2"][10] = -1

    edge_channels = (0, 11, 13, 31)
    for dst, src in zip(edge_channels, (24, 25, 26, 27)):
        c2(dst, src)

    for src in (3, 4, 5):
        c2(12, src)

    c2(14, 1)
    c2(15, 11)
    c2(16, 12)

    for o in (17, 18, 19):
        c2(o, 28, 0.5, 0, -1)
        c2(o, 28, 0.5, 0, 1)
    c2(17, 2)
    c2(17, 2, 1, 1, 0)
    p["b2"][17] = -2
    c2(18, 1)
    p["b2"][18] = -1
    c2(19, 2)
    c2(19, 1, 1, 1, 0)
    p["b2"][19] = -2

    for o in (20, 21, 22):
        c2(o, 29, 0.5, -1, 0)
        c2(o, 29, 0.5, 1, 0)
    c2(20, 2)
    c2(20, 2, 1, 0, -1)
    p["b2"][20] = -2
    c2(21, 1)
    p["b2"][21] = -1
    c2(22, 2)
    c2(22, 1, 1, 0, -1)
    p["b2"][22] = -2

    for j in range(4):
        c2(23 + j, 16 + j)

    # Avoid borrowing the occupancy of another upper compartment.
    c2(25, 22, 1, -1, 0)
    p["b2"][25] = -1

    c2(29, 7, 1, 1, 0)
    c2(29, 0, -8)

    for src in (3, 5, 7, 13):
        c2(30, src)

    # ---------------------------------------------------------
    # Axial layer.
    # ---------------------------------------------------------
    f3(0, 30)
    f3(0, 3, -1)
    f3(0, 15, -1)
    f3(0, 16, -1)
    f3(1, 3)

    f3(2, 5)
    f3(3, 7)
    f3(4, 9)

    for dst, src in ((5, 17), (6, 18), (7, 19)):
        f3(dst, src)
        f3(dst, 5, 1, 1)
        p["b3"][dst] = -1

    for dst, src in ((8, 20), (9, 21), (10, 22)):
        f3(dst, src)
        f3(dst, 7, 1, 2)
        p["b3"][dst] = -1

    f3(11, 12)

    f3(12, 1)
    for src in edge_channels:
        f3(12, src, -1)

    for j in range(4):
        f3(13 + j, 23 + j)
        if j != 2:
            f3(13 + j, 14, 1, 2 if j < 2 else 1)
            p["b3"][13 + j] = -1

    p["b3"][17] = 1

    # Ready gate endpoints on the actor's foot coordinate.
    for dst, endpoint, foot, pool in (
        (18, 15, 5, 1),
        (19, 16, 7, 2),
    ):
        f3(dst, endpoint)
        f3(dst, 2)
        f3(dst, foot, 1, pool)
        p["b3"][dst] = -2

    # Prefer the upper approach to goals on horizontal gate walls.
    for dst, source in ((20, 3), (21, 29)):
        f3(dst, source)
        f3(dst, 27, 1, 1)
        f3(dst, 10, -1, 1)
        p["b3"][dst] = -1

    # Pressure barriers, with openings at transfer-button pairs.
    f3(22, 30)
    f3(22, 0, -0.5, 1)
    f3(22, 11, -0.5, 1)
    f3(22, 13, -0.5, 2)
    f3(22, 31, -0.5, 2)

    f3(23, 12)
    f3(23, 0, 1, 1)
    f3(23, 11, 1, 1)
    p["b3"][23] = -2

    f3(24, 12)
    f3(24, 13, 1, 2)
    f3(24, 31, 1, 2)
    p["b3"][24] = -2

    f3(25, 30)

    # Pairs whose differences are binary:
    # 26-27: actor is right of a ready vertical gate.
    # 28-29: actor is below a ready horizontal gate.
    for a, b, endpoint, foot, pool, actor_coord, coord in (
        (26, 27, 15, 5, 1, 4, 6),
        (28, 29, 16, 7, 2, 28, 8),
    ):
        for dst, offset in ((a, 0.5), (b, -0.5)):
            f3(dst, endpoint, 256)
            f3(dst, 2, 256)
            f3(dst, foot, 256, pool)
            f3(dst, actor_coord, 63, pool)
            f3(dst, coord, -63)
            p["b3"][dst] = -768 + offset

    # ---------------------------------------------------------
    # Recurrent reachability and arrival-time field.
    # ---------------------------------------------------------
    neighborhood = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    for o in (0, 1):
        for dr, dc in neighborhood:
            rec(o, 0, 1, dr, dc)
            rec(o, 1, -1, dr, dc)
        p["Uh"][o, 0] = -16
        p["Uh"][o, 1] = 1
        p["Uh"][o, 20] = -17
        p["Uh"][o, 21] = 1
    p["bh"][1] = -1

    rec(2, 2)
    rec(2, 0)
    rec(2, 1, -1)

    force_specs = (
        (3, 2, -1, 0), (4, 2, 1, 0),
        (5, 3, 0, 1), (6, 3, 0, -1),
        (7, 4, 0, -1), (8, 4, 0, 1),
    )
    for o, mask, dr, dc in force_specs:
        rec(o, 2, 1, dr, dc)
        rec(o, 2, -1, -dr, -dc)
        p["Uh"][o, mask] = 256
        p["bh"][o] = -257.25

    # Reuse directional derivatives at ready gates, enabling only
    # the direction away from the actor.
    gate_base = 8.0
    gate_enable = 257.25 + gate_base
    enabled = {
        3: {28: 1, 29: -1},
        4: {19: 1, 28: -1, 29: 1},
        5: {18: 1, 26: -1, 27: 1},
        6: {26: 1, 27: -1},
    }
    for channel, terms in enabled.items():
        for feature, coefficient in terms.items():
            p["Uh"][channel, feature] += gate_enable * coefficient

    # Grow and drain pressure fields.
    for o in (9, 10):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rec(o, o, 1.0 / 9.0, dr, dc)
        p["Uh"][o, 22] = -64

    for src in (3, 5, 7):
        rec(9, src, 0.25)
    for src in (4, 6, 8):
        rec(10, src, 0.25)

    # Remove the fixed gate score from pressure injection.
    for pressure, sources in ((9, (3, 5)), (10, (4, 6))):
        for source in sources:
            for feature, coefficient in enabled[source].items():
                p["Uh"][pressure, feature] -= (
                    0.25 * gate_base * coefficient
                )

    p["Uh"][9, 5] += 2
    p["Uh"][9, 8] += 2
    p["Uh"][10, 6] += 2
    p["Uh"][10, 9] += 2

    # Protect compartments already aligned with closed gates.
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rec(15, 15, 1.0 / 9.0, dr, dc)
    p["Uh"][15, 7] = 2
    p["Uh"][15, 10] = 2
    p["Uh"][15, 25] = -64

    # Donor-valid pressure-difference relays.
    carrier = 4.0
    for j, (dr, dc) in enumerate(directions):
        o = 11 + j
        pr, pc = -dr, -dc

        rec(o, o, 1, pr, pc)
        rec(o, 9, 1)
        rec(o, 10, -1)
        rec(o, 9, -1, pr, pc)
        rec(o, 10, 1, pr, pc)
        rec(o, 15, -8)
        rec(o, 15, -8, pr, pc)

        p["Uh"][o, 13 + j] = carrier
        p["Uh"][o, 23 if j < 2 else 24] = 128
        p["bh"][o] = -128

    # ---------------------------------------------------------
    # Click decoding.
    # ---------------------------------------------------------
    p["bc"][0] = -gate_base
    p["Wc"][0, 12] = gate_base - carrier

    for j in range(4):
        p["Wc"][0, 13 + j] = -carrier
        p["Wc"][0, 32 + 11 + j] = 1

    for channel in (3, 4, 5, 6):
        p["Wc"][0, 32 + channel] = 1

    # Directional derivatives at feet are control signals, not
    # clickable targets.
    for feature in (2, 3, 4):
        p["Wc"][0, feature] = -256

    p["bo"][6] = 1
    return p
