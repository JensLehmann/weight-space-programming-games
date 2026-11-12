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

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    sides = [(0, 1), (0, 1), (1, 0), (1, 0)]

    def c1(o, c, r=0, s=0, v=1.0):
        p["W1"][o, c, r + 2, s + 2] += v

    def c2(o, c, r=0, s=0, v=1.0):
        p["W2"][o, c, r + 1, s + 1] += v

    def c3(o, c, v=1.0, axis=None):
        offset = 0 if axis is None else 32 if axis == "row" else 64
        p["W3"][o, offset + c] += v

    def rh(o, c, r=0, s=0, v=1.0):
        p["Wh"][o, c, r + 1, s + 1] += v

    # Palette primitives: the selected piece is provisionally traversable.
    for color in [12, 9, 6]:
        c1(0, color)
    for color in [8, 15]:
        c1(1, color)
    c1(2, 9)
    for color in [11, 13]:
        c1(3, color)
    c1(4, 4)
    c1(5, 6)

    # Cup material two pixels away, extended to radius three by W2.
    for k, (dr, dc) in enumerate(dirs):
        for color in [11, 13]:
            c1(6 + k, color, 2 * dr, 2 * dc)

    # Shifted selected-colour planes.
    shifted = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    for k, (dr, dc) in enumerate(shifted):
        c1(10 + k, 9, dr, dc)

    # Selected exterior boundary masks.
    c1(14, 9, 0, -1)
    c1(14, 9, 0, 1)
    c1(14, 9, v=-2.0)
    c1(15, 9, -1, 0)
    c1(15, 9, 1, 0)
    c1(15, 9, v=-2.0)

    # Unselected interior boundary masks.
    for color in [8, 15]:
        c1(16, color, v=2.0)
        c1(16, color, 0, -1, -1.0)
        c1(16, color, 0, 1, -1.0)
        c1(17, color, v=2.0)
        c1(17, color, -1, 0, -1.0)
        c1(17, color, 1, 0, -1.0)

    c1(18, 16)
    c1(19, 17)

    for k, (dr, dc) in enumerate(dirs):
        c1(20 + k, 4)
        c1(20 + k, 6, 2 * dr, 2 * dc)
        p["b1"][20 + k] = -1.0

    # Long-axis witnesses distinguish thin bars from compact pieces.
    for out, colors, horizontal in [
        (24, [9], True),
        (25, [9], False),
        (26, [8, 15], True),
        (27, [8, 15], False),
    ]:
        for color in colors:
            c1(out, color,
               0 if horizontal else -2,
               -2 if horizontal else 0)
            c1(out, color,
               0 if horizontal else 2,
               2 if horizontal else 0)
        p["b1"][out] = -1.0

    c1(28, 1)
    c1(29, 18)

    # ---------------------------------------------------------------
    # Second convolution.
    # ---------------------------------------------------------------
    for ch in [0, 1, 2]:
        c2(ch, ch)

    # Cup openings: background flanked by prongs, with a base forward.
    for k, ((dr, dc), (tr, tc)) in enumerate(zip(dirs, sides)):
        out = 3 + k
        c2(out, 0)
        c2(out, 6 + k, dr, dc)
        kp = dirs.index((tr, tc))
        km = dirs.index((-tr, -tc))
        c2(out, 6 + kp, tr, tc)
        c2(out, 6 + km, -tr, -tc)
        p["b2"][out] = -3.0

    # Candidate endpoint masks shifted two pixels. The recurrent
    # readout adds another pixel of displacement.
    for out, plane, horizontal in [
        (7, 13, True),   # left candidate
        (8, 12, True),   # right candidate
        (9, 11, False),  # up candidate
        (10, 10, False), # down candidate
    ]:
        if horizontal:
            c2(out, plane, 0, -1)
            c2(out, plane, 0, 1)
        else:
            c2(out, plane, -1, 0)
            c2(out, plane, 1, 0)
        c2(out, plane, v=-2.0)

    c2(11, 14)  # current horizontal endpoints
    c2(12, 15)  # current vertical endpoints
    c2(13, 16)
    c2(14, 17)
    c2(15, 24)
    c2(16, 25)

    # Selected-coordinate extrema.
    for out, coord, sign, bias in [
        (17, 18, 1.0, -1.0),
        (18, 19, 1.0, -1.0),
        (19, 18, -1.0, 0.0),
        (20, 19, -1.0, 0.0),
    ]:
        c2(out, 2)
        c2(out, coord, v=sign)
        p["b2"][out] = bias

    for k in range(4):
        c2(21 + k, 20 + k)

    c2(25, 26)
    c2(26, 27)
    c2(27, 18)
    c2(28, 19)
    c2(29, 28)

    # ---------------------------------------------------------------
    # Axial features.
    # ---------------------------------------------------------------
    for ch in range(13):
        c3(ch, ch)

    # Only boundaries along a piece's long axis generate deficits.
    c3(13, 13)
    c3(13, 25, axis="row")
    p["b3"][13] = -1.0

    c3(14, 14)
    c3(14, 26, axis="col")
    p["b3"][14] = -1.0

    for ch in range(15, 25):
        c3(ch, ch)

    c3(25, 25)
    c3(26, 26)
    c3(29, 27)
    c3(30, 28)
    c3(31, 29)

    # ---------------------------------------------------------------
    # Recurrent fields:
    # 0..3 straight backward cup fields
    # 4..7 laterally smoothed placement potentials
    # 8..11 candidate endpoint scores
    # 12..13 current endpoint scores
    # 14..15 unselected-piece connection deficits
    # ---------------------------------------------------------------
    gate = 10.0

    for k, ((dr, dc), (tr, tc)) in enumerate(zip(dirs, sides)):
        beam = k
        lateral = 4 + k

        rh(beam, beam, dr, dc, 0.985)
        p["Uh"][beam, 0] = gate
        p["Uh"][beam, 3 + k] = 1.0
        p["bh"][beam] = -gate

        rh(lateral, lateral, tr, tc, 0.48)
        rh(lateral, lateral, -tr, -tc, 0.48)
        rh(lateral, beam, v=0.025)
        p["Uh"][lateral, 0] = gate
        p["Uh"][lateral, 1] = gate
        p["bh"][lateral] = -gate

    for out, mask, dr, dc, fields in [
        (8, 7, 0, -1, [4, 5]),
        (9, 8, 0, 1, [4, 5]),
        (10, 9, -1, 0, [6, 7]),
        (11, 10, 1, 0, [6, 7]),
        (12, 11, 0, 0, [4, 5]),
        (13, 12, 0, 0, [6, 7]),
    ]:
        for field in fields:
            rh(out, field, dr, dc)
        p["Uh"][out, mask] = gate
        p["bh"][out] = -gate

    # Deficits reside inside unselected pieces, so their maxima are
    # valid click targets rather than background pixels.
    for out, mask, fields, offsets in [
        (14, 13, [4, 5], [(0, -1), (0, 1)]),
        (15, 14, [6, 7], [(-1, 0), (1, 0)]),
    ]:
        p["Uh"][out, mask] = gate + 0.10
        p["bh"][out] = -gate
        for field in fields:
            for dr, dc in offsets:
                rh(out, field, dr, dc, -0.5)

    # ---------------------------------------------------------------
    # Candidate-minus-current comparisons, using spatial sums.
    # ---------------------------------------------------------------
    for unit, candidate, current in [
        (0, 8, 12), (1, 9, 12),
        (2, 10, 13), (3, 11, 13),
    ]:
        p["Wz"][unit, 16 + candidate] = 4096.0
        p["Wz"][unit, 16 + current] = -4096.0
        p["bz"][unit] = -0.025

    # Axis penalties vanish for compact pieces with both witnesses.
    p["Wz"][4, 32 + 16] = 1.0
    p["Wz"][4, 32 + 15] = -1.0
    p["Wz"][5, 32 + 15] = 1.0
    p["Wz"][5, 32 + 16] = -1.0

    threshold = 1.0 - 0.5 / 63.0
    for unit, feature in [
        (6, 20), (7, 18), (8, 19), (9, 17)
    ]:
        p["Wz"][unit, 32 + feature] = 1.0
        p["bz"][unit] = -threshold

    p["Wz"][10, 14] = 1.0
    p["Wz"][11, 15] = 1.0
    p["Wz"][12, 32 + 1] = 1.0

    # Bounded selection-cycle counter.
    p["Wm"][0, 0] = 1.0
    p["Wma"][0, 6] = 0.18
    p["Wma"][0, 5] = -0.8
    p["Wm"][1, 0] = 20.0
    p["bm"][1] = -12.0

    # Previous-action indicators for mild reversal hysteresis.
    for action in [1, 2, 3, 4]:
        p["Wma"][8 + action, action] = 2.0

    # ---------------------------------------------------------------
    # Action and click heads.
    # ---------------------------------------------------------------
    p["bo"][:] = -1000.0

    for action, score in [(3, 0), (4, 1), (1, 2), (2, 3)]:
        p["Wo"][action, score] = 1.0
        p["bo"][action] = -0.005

    for action in [3, 4]:
        p["Wo"][action, 4] = -20.0
    for action in [1, 2]:
        p["Wo"][action, 5] = -20.0

    for action, unit in [(3, 6), (4, 7), (1, 8), (2, 9)]:
        p["Wo"][action, unit] = -4000.0

    for action, opposite in [(1, 2), (2, 1), (3, 4), (4, 3)]:
        p["Wo"][action, 128 + 8 + opposite] = -0.01

    # No positive click logit exists when there is no unselected piece.
    p["bo"][6] = -0.005
    p["Wo"][6, 10] = 0.10
    p["Wo"][6, 11] = 0.10

    # At episode start this is approximately .001. After repeated
    # selection cycles it rises enough to trigger a spill attempt.
    p["bo"][5] = 0.051
    p["Wo"][5, 128 + 1] = 0.05

    p["Wc"][0, 1] = 10.0
    p["Wc"][0, 32 + 14] = 10.0
    p["Wc"][0, 32 + 15] = 10.0

    return p
