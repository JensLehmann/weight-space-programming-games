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
    p = {k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()}

    W1, b1 = p["W1"], p["b1"]
    W2, b2 = p["W2"], p["b2"]
    W3, b3 = p["W3"], p["b3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]
    Wz, bz = p["Wz"], p["bz"]
    Wm, Wmz, Wma, bm = p["Wm"], p["Wmz"], p["Wma"], p["bm"]
    Wo, bo = p["Wo"], p["bo"]

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    actions = [1, 2, 3, 4]
    switch_colors = [12, 14, 15]

    def c1(o, i, v=1.0, dy=0, dx=0):
        W1[o, i, dy + 2, dx + 2] += v

    def c2(o, i, v=1.0, dy=0, dx=0):
        W2[o, i, dy + 1, dx + 1] += v

    def cr(o, i, v=1.0, dy=0, dx=0):
        Wh[o, i, dy + 1, dx + 1] += v

    # ------------------------------------------------------------
    # Visual layer 1.
    # ------------------------------------------------------------
    c1(0, 10)

    # Colour-1 neighbourhood indicates selected-box control mode.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            c1(1, 1, dy=dy, dx=dx)

    # Small box glyph interiors, rejecting broad colour-11 walls.
    for channel, color in [(2, 9), (3, 11)]:
        c1(channel, color)
        for dy, dx in directions:
            c1(channel, color, -1.0, 2 * dy, 2 * dx)

    c1(4, 16)
    c1(5, 17)

    for color in (5, 10, 1):
        c1(6, color)

    for j, color in enumerate(switch_colors):
        c1(7 + j, color)
        for d, (dy, dx) in enumerate(directions):
            c1(10 + 4 * j + d, color, dy=2 * dy, dx=2 * dx)

        # Wall-palette samples at the display boundaries.
        c1(22 + 2 * j, color)
        c1(22 + 2 * j, 17, -64.0)
        c1(23 + 2 * j, color)
        c1(23 + 2 * j, 17, 64.0)
        b1[23 + 2 * j] = -64.0

    # Floor/switch samples three pixels beyond a token edge, after W2.
    for d, (dy, dx) in enumerate(directions):
        for color in (5, 10, 1, 12, 14, 15):
            c1(28 + d, color, dy=2 * dy, dx=2 * dx)

    # ------------------------------------------------------------
    # Visual layer 2.
    # ------------------------------------------------------------
    c2(0, 0)

    # Active group tokens or the selected box, never inactive tokens.
    c2(1, 0)
    c2(1, 3)

    # Expand outlined box interiors to their floor-coloured borders.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            c2(2, 2, dy=dy, dx=dx)
            c2(2, 3, dy=dy, dx=dx)

    c2(3, 3)
    c2(4, 4)
    c2(5, 5)
    c2(6, 6)

    # Isolated coloured squares: colour is absent three pixels away
    # in every cardinal direction.
    for j in range(3):
        c2(7 + j, 7 + j)
        for d, (dy, dx) in enumerate(directions):
            c2(7 + j, 10 + 4 * j + d, -1.0, dy, dx)

        c2(10 + j, 22 + 2 * j)
        c2(10 + j, 23 + 2 * j)

    c2(13, 5)
    b2[13] = -0.5
    c2(14, 5, -1.0)
    b2[14] = 0.5

    c2(15, 6)
    c2(16, 2)

    for d, (dy, dx) in enumerate(directions):
        c2(17 + d, 0)
        c2(17 + d, 3)
        c2(17 + d, 0, -1.0, dy, dx)
        c2(17 + d, 3, -1.0, dy, dx)
        c2(22 + d, 28 + d, dy=dy, dx=dx)

    c2(21, 3)
    c2(26, 1)

    for j in range(3):
        c2(27 + j, 7 + j)

    c2(30, 4, -64.0)
    b2[30] = 1.0
    c2(31, 4, 64.0)
    b2[31] = -63.0

    # ------------------------------------------------------------
    # Axial layer.
    # ------------------------------------------------------------
    # Clipped union of ordinary floor and optimistic box routing.
    W3[0, 6] = 1.0
    W3[0, 2] = 1.0
    W3[13, 6] = 1.0
    W3[13, 2] = 1.0
    b3[13] = -1.0

    # Fixed passable midline target. No moving-token self-target.
    W3[1, 15] = 1.0
    W3[1, 2] = 1.0
    W3[1, 13] = -63.0
    W3[1, 14] = -63.0
    W3[1, 30] = -512.0
    W3[1, 31] = -512.0

    for j in range(3):
        # Switch sources, excluding any matching wall palette.
        W3[2, 7 + j] = 1.0
        W3[2, 32 + 10 + j] = -1.0

        # Complete switch square from its isolated inner pixels.
        # Long gates lack the required row/column intersection.
        out = 16 + j
        W3[out, 27 + j] = 1.0
        W3[out, 32 + 7 + j] = 1.0
        W3[out, 64 + 7 + j] = 1.0
        W3[out, 32 + 10 + j] = -2.0
        b3[out] = -2.0

    W3[3, 1] = 1.0
    W3[3, 13] = -64.0
    W3[8, 1] = 1.0
    W3[8, 14] = -64.0

    for d in range(4):
        # Exposed edges with a plausible adjacent-cell sample.
        for out, half in [(4 + d, 13), (9 + d, 14)]:
            W3[out, 17 + d] = 1.0
            W3[out, 22 + d] = 1.0
            W3[out, half] = -64.0
            b3[out] = -1.0

    W3[14, 21] = 1.0

    # Ordinary-floor-only box clearance flags.
    for d in range(4):
        W3[19 + d, 21] = 1.0
        W3[19 + d, 22 + d] = 1.0
        b3[19 + d] = -1.0

    W3[23, 26] = 1.0

    for out, coordinate, excluded in [
        (24, 5, 13),
        (25, 5, 14),
        (26, 4, 13),
        (27, 4, 14),
    ]:
        W3[out, 0] = 10.0
        W3[out, coordinate] = 1.0
        W3[out, excluded] = -512.0
        b3[out] = -10.0

    W3[29, 30] = 1.0
    W3[29, 31] = 1.0

    # Proximity of unselected boxes to active group tokens.
    W3[30, 16] = 64.0
    for d in range(4):
        W3[30, 32 + 17 + d] = 0.25
    b3[30] = -64.0

    W3[31, 16] = 1.0

    # ------------------------------------------------------------
    # Recurrent distance fields.
    #
    # A-B clips a flood to [0,1]; C integrates visitation.
    # Closed gates are absent from the traversable mask.
    # ------------------------------------------------------------
    for base, source in [(0, 1), (3, 2)]:
        for dy, dx in [(0, 0)] + directions:
            for out in (base, base + 1):
                cr(out, base, 1.0, dy, dx)
                cr(out, base + 1, -1.0, dy, dx)

        for out in (base, base + 1):
            Uh[out, 0] = 512.0
            Uh[out, 13] = -512.0
            for j in range(3):
                Uh[out, 16 + j] = 512.0
            Uh[out, 29] = -512.0
            Uh[out, source] = 1.0
            bh[out] = -512.0

        bh[base + 1] -= 1.0
        cr(base + 2, base + 2)
        cr(base + 2, base)
        cr(base + 2, base + 1, -1.0)

    gradient_offset = 8.0
    switch_weight = 1.5

    for first, edge_first in [(6, 4), (10, 9)]:
        for d, (dy, dx) in enumerate(directions):
            out = first + d
            cr(out, 2, 1.0, dy, dx)
            cr(out, 2, -1.0)
            cr(out, 5, switch_weight, dy, dx)
            cr(out, 5, -switch_weight)
            Uh[out, edge_first + d] = 512.0
            bh[out] = gradient_offset - 512.0

    # Additional local heat for the click map.
    cr(14, 2)
    Uh[14, 31] = 512.0
    bh[14] = -512.0

    # ------------------------------------------------------------
    # Pooled signed gradients.
    # ------------------------------------------------------------
    def hm(out, channel, value=1.0):
        Wz[out, channel] += value

    def fm(out, channel, value=1.0):
        Wz[out, 32 + channel] += value

    for group, first, edge_first in [(0, 6, 4), (1, 10, 9)]:
        for d in range(4):
            i = 4 * group + d
            hm(i, first + d)
            fm(i, edge_first + d, -gradient_offset)
            hm(8 + i, first + d, -1.0)
            fm(8 + i, edge_first + d, gradient_offset)

    fm(18, 23)
    fm(19, 23)
    bz[19] = -1.0

    fm(20, 30)
    fm(29, 30)
    bz[29] = -1.0

    for d in range(4):
        fm(21 + d, 19 + d)

    for z, channel in [(25, 24), (26, 25), (27, 26), (28, 27)]:
        fm(z, channel)

    # ------------------------------------------------------------
    # Memory and action selection.
    # ------------------------------------------------------------
    n = 0

    def allocate():
        nonlocal n
        u = n
        n += 1
        return u

    def mode(unit, coefficient):
        Wmz[unit, 18] += coefficient
        Wmz[unit, 19] -= coefficient

    coords = [allocate() for _ in range(4)]
    coord_features = [25, 26, 27, 28]
    for u, z in zip(coords, coord_features):
        Wmz[u, z] = 0.01

    OX, OY = allocate(), allocate()

    def delta(unit, index, coefficient):
        Wmz[unit, coord_features[index]] += 0.01 * coefficient
        Wm[unit, coords[index]] -= coefficient

    # Reflection-sign calibration.
    for orientation, index, positive, negative in [
        (OX, 0, 4, 3),
        (OY, 2, 2, 1),
    ]:
        Wm[orientation, orientation] = 2.0
        bm[orientation] = 0.05

        for action, sign in [(positive, 1.0), (negative, -1.0)]:
            plus, minus = allocate(), allocate()
            delta(plus, index, 20000.0 * sign)
            delta(minus, index, -20000.0 * sign)

            for u in (plus, minus):
                Wma[u, action] = 1000.0
                bm[u] = -1000.0
                mode(u, -2000.0)

            Wm[orientation, plus] += 2.0
            Wm[orientation, minus] -= 2.0

    # Signed preferences remain negative when a move is counterproductive.
    for group in range(2):
        for d in range(4):
            i = 4 * group + d
            for inverted in (False, True):
                u = allocate()
                orientation = OY if d < 2 else OX
                Wmz[u, i] = 0.65
                Wmz[u, 8 + i] = -0.65
                Wm[u, orientation] = -12.0 if inverted else 12.0
                bm[u] = -12.0
                mode(u, -24.0)

                physical = d ^ int(inverted)
                if group == 1 and d >= 2:
                    physical ^= 1
                action = actions[physical]
                Wo[action, 128 + u] += 5.0
                bo[action] += 5.0

    # Ordinary, unreflected box moves, biased away from the junction.
    for d in range(4):
        u = allocate()
        for group in range(2):
            i = 4 * group + d
            Wmz[u, 8 + i] = 0.35
            Wmz[u, i] = -0.35
        mode(u, 12.0)
        Wmz[u, 21 + d] = 5.0
        bm[u] = -17.0
        action = actions[d]
        Wo[action, 128 + u] = 5.0
        bo[action] += 5.0

    # Coordinate equality tests.
    tests = []
    for index in range(4):
        for sign in (1.0, -1.0):
            u = allocate()
            tests.append(u)
            delta(u, index, 8000.0 * sign)
            bm[u] = 1.0

    previous = []
    for action in actions:
        u = allocate()
        previous.append(u)
        Wma[u, action] = 4.0
        bm[u] = -2.0

    # Stronger command-specific no-motion penalties.
    for d, action in enumerate(actions):
        u = allocate()
        for test in tests:
            Wm[u, test] = 3.0
        Wm[u, previous[d]] = 12.0
        mode(u, -16.0)
        bm[u] = -28.0
        Wo[action, 128 + u] = -10.0
        bo[action] -= 10.0

    # Penalize immediate reversal.
    for d, u in enumerate(previous):
        opposite = actions[d ^ 1]
        Wo[opposite, 128 + u] -= 1.0
        bo[opposite] -= 1.0

    # Detect alternating opposite commands using the delayed one-hot.
    for d, action in enumerate(actions):
        u = allocate()
        Wm[u, previous[d]] = 4.0
        Wma[u, actions[d ^ 1]] = 4.0
        bm[u] = -6.0
        Wo[action, 128 + u] -= 3.0
        bo[action] -= 3.0

    # Box timer resets as soon as group control resumes.
    H1, H2, H3 = allocate(), allocate(), allocate()
    mode(H1, 6.0)
    bm[H1] = -3.0

    for out, preceding in [(H2, H1), (H3, H2)]:
        Wm[out, preceding] = 4.0
        mode(out, 8.0)
        bm[out] = -8.0

    select, leave = allocate(), allocate()

    for test in tests:
        Wm[select, test] = 4.0
    Wmz[select, 20] = 9.0
    Wmz[select, 29] = -9.0
    mode(select, -32.0)
    bm[select] = -31.0

    mode(leave, 8.0)
    Wm[leave, H3] = 6.0
    bm[leave] = -10.0

    Wo[6, 128 + select] = 30.0
    Wo[6, 128 + leave] = 30.0
    bo[6] = 30.0

    # Weak nonrepeating tie perturbation, not a level-specific sequence.
    u, v = allocate(), allocate()
    angle = 0.37
    c, s = math.cos(angle), math.sin(angle)
    Wm[u, u] = 1.015 * c
    Wm[u, v] = -1.015 * s
    Wm[v, u] = 1.015 * s
    Wm[v, v] = 1.015 * c
    bm[u] = 0.003
    bm[v] = 0.002
    Wo[1, 128 + u] = 0.25
    Wo[2, 128 + u] = -0.25
    Wo[3, 128 + v] = 0.25
    Wo[4, 128 + v] = -0.25

    assert n <= 64

    bo[1] += 0.004
    bo[2] += 0.003
    bo[3] += 0.002
    bo[4] += 0.001

    # Prefer outward/right clearance only as a box-mode tie-break.
    Wo[4, 128 + H1] += 0.4
    bo[4] += 0.4

    bo[0] = -1000.0
    bo[5] = -1000.0
    bo[7] = -1000.0

    # Box interior wins ordinary-mode clicks. In box mode the broad
    # colour-1 neighbourhood wins, restoring reflected group control.
    p["Wc"][0, 31] = 1000.0
    p["Wc"][0, 30] = 100.0
    p["Wc"][0, 23] = 10000.0
    p["Wc"][0, 32 + 14] = 0.5

    return p
