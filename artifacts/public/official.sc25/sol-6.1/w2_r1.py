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
    p = {k: np.zeros(s, np.float32) for k, s in shapes.items()}
    W1, b1 = p["W1"], p["b1"]
    W2, b2 = p["W2"], p["b2"]
    W3, b3 = p["W3"], p["b3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    def c1(o, i, y=0, x=0, v=1.0):
        W1[o, i, y + 2, x + 2] += v

    def c2(o, i, y=0, x=0, v=1.0):
        W2[o, i, y + 1, x + 1] += v

    def rec(o, i, y=0, x=0, v=1.0):
        Wh[o, i, y + 1, x + 1] += v

    palette = (0, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15)
    ci = {color: i for i, color in enumerate(palette)}
    for color, channel in ci.items():
        c1(channel, color)

    small = 12
    large = (13, 14, 15, 16)
    door_h, door_v = 17, 18
    no_gate, portal_count, no_portal = 19, 20, 21
    size_icon, tele_icon, fire_icon = 22, 23, 24
    block2, pad, row, col, free4 = 25, 26, 27, 28, 29

    # Small player: an isolated 2x2 patch of player colours.
    for y in range(-1, 3):
        for x in range(-1, 3):
            inside = 0 <= y <= 1 and 0 <= x <= 1
            for color in (9, 10):
                c1(small, color, y, x, 2.0 if inside else -2.0)
    b1[small] = -7.0

    # Large player: four orientations, anchored at top-left+(1,1).
    template = np.array([
        [10, 10, 10, 10],
        [10, 10, 10, 10],
        [9, 9, 9, 9],
        [9, 9, 9, 9],
    ])
    for rotation, out in enumerate(large):
        t = np.rot90(template, rotation)
        for y in range(4):
            for x in range(4):
                c1(out, int(t[y, x]), y - 1, x - 1, 2.0)
        b1[out] = -31.0

    for d in range(-2, 3):
        c1(door_h, 9, 0, d, 2.0)
        c1(door_v, 9, d, 0, 2.0)
    b1[door_h] = b1[door_v] = -9.0

    b1[no_gate] = 1.0
    for y in range(-1, 2):
        for x in range(-1, 2):
            for color in (12, 13):
                c1(no_gate, color, y, x, -1.0)

    b1[no_portal] = 1.0
    for y in range(-2, 3):
        for x in range(-2, 3):
            c1(portal_count, 11, y, x)
            c1(no_portal, 11, y, x, -1.0)

    # Require the icon's top 2x2 patch and its panel border.
    # This excludes similarly coloured objects in the playfield.
    for out, color in (
        (size_icon, 15), (tele_icon, 11), (fire_icon, 6)
    ):
        for y in (0, 1):
            for x in (0, 1):
                c1(out, color, y, x, 2.0)
        c1(out, 3, -1, 0, 2.0)
        b1[out] = -9.0

    obstacles = (3, 4, 5, 6, 12, 13)
    for y in (0, 1):
        for x in (0, 1):
            for color in obstacles:
                c1(block2, color, y, x)

    b1[free4] = 1.0
    for y in range(-1, 3):
        for x in range(-1, 3):
            for color in obstacles:
                c1(free4, color, y, x, -1.0)

    for y in range(-2, 3):
        for x in range(-2, 3):
            if abs(y) <= 1 and abs(x) <= 1:
                for color in (0, 2, 14):
                    c1(pad, color, y, x, 2.0)
            else:
                c1(pad, 3, y, x, 2.0)
    b1[pad] = -49.0
    c1(row, 16)
    c1(col, 17)

    # F2: player flags at the actual top-left.
    c2(0, small)
    c2(2, small)
    for channel in large:
        for out in (1, 2, 3, 30):
            c2(out, channel, 1, 1)

    # Large player away from a portal.
    c2(3, no_portal)
    b2[3] = -1.0

    # Small player on a portal, with room to grow in place.
    c2(4, small)
    c2(4, free4, 1, 1)
    c2(4, no_portal, v=-1.0)
    b2[4] = -1.0

    # Small player away from a portal.
    c2(5, small)
    c2(5, no_portal)
    b2[5] = -1.0
    c2(30, no_portal, v=-1.0)

    for y in range(-1, 2):
        for x in range(-1, 2):
            c2(6, door_h, y, x)
            c2(6, door_v, y, x)

    c2(7, ci[6])
    c2(7, no_gate, v=-1.0)
    c2(8, block2)
    for color in (3, 5, 12, 13):
        c2(9, ci[color])

    # Facing flags: up, down, left, right.
    small_faces = (
        ((9, 0, 0), (10, 1, 0)),
        ((10, 0, 0), (9, 1, 0)),
        ((9, 0, 0), (10, 0, 1)),
        ((10, 0, 0), (9, 0, 1)),
    )
    large_faces = (15, 13, 16, 14)
    for j, entries in enumerate(small_faces):
        out = 10 + j
        c2(out, small, v=10.0)
        for color, y, x in entries:
            c2(out, ci[color], y, x)
        c2(out, large_faces[j], 1, 1, 12.0)
        b2[out] = -11.0

    c2(14, size_icon)
    c2(15, tele_icon)
    c2(16, fire_icon)

    for out, coordinate in ((17, row), (18, col)):
        c2(out, coordinate, v=0.02)
        c2(out, small)
        for channel in large:
            c2(out, channel, 1, 1)
        b2[out] = -1.0

    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for j, (dy, dx) in enumerate(directions):
        # Legal small-player neighbour, anchored at its top-left.
        out = 19 + j
        c2(out, small, -dy, -dx)
        c2(out, block2, v=-1.0)

        # Legal large-player neighbour, anchored at top-left+(1,1).
        out = 23 + j
        for channel in large:
            c2(out, channel, -dy, -dx)
        c2(out, free4)
        b2[out] = -1.0

    c2(27, pad)
    c2(27, ci[14])
    b2[27] = -1.0
    c2(28, pad)
    c2(28, ci[0])
    b2[28] = -1.0
    c2(29, pad)

    # F3: local features; the axial inputs are unused.
    for channel in range(19):
        W3[channel, channel] = 1.0

    for j in range(4):
        W3[19 + j, 19 + j] = 1.0
        W3[19 + j, 23 + j] = 1.0

    W3[23, 3] = 1.0
    W3[23, 4] = 1.0             # size command source
    W3[24, 5] = 1.0
    W3[24, 30] = 1.0            # teleport command source
    W3[25, 27] = 1.0            # active grid cell
    W3[26, 28] = 1.0            # highlighted empty grid cell
    W3[27, 3] = 1.0             # automatic shrinking candidate
    W3[28, 30] = 1.0
    W3[31, 29] = 1.0

    # H0/H1: horizontal and vertical firing-line potentials.
    rec(0, 0, 0, -1, 0.5)
    rec(0, 0, 0, 1, 0.5)
    rec(1, 1, -1, 0, 0.5)
    rec(1, 1, 1, 0, 0.5)
    for out in (0, 1):
        Uh[out, 7] = 1.0
        Uh[out, 9] = -10000.0

    # H2-H3: bounded flood-fill from doors and firing opportunities.
    neighbours = ((0, 0),) + directions
    for y, x in neighbours:
        for out in (2, 3):
            rec(out, 2, y, x)
            rec(out, 3, y, x, -1.0)
    for out in (2, 3):
        rec(out, 0, v=100.0)
        rec(out, 1, v=100.0)
        Uh[out, 6] = 1.0
        Uh[out, 8] = -1.0e7
    bh[3] = -1.0

    # H4: accumulated arrival-time potential.
    rec(4, 4)
    rec(4, 2)
    rec(4, 3, v=-1.0)

    # H5..H8: independent, legality-gated directional readouts.
    for j in range(4):
        out = 5 + j
        rec(out, 4)
        rec(out, 0 if j >= 2 else 1, v=5.0)
        Uh[out, 19 + j] = 20000.0
        bh[out] = -10000.0

    # H15: firing potential specifically at the player's top-left.
    rec(15, 0, v=100.0)
    rec(15, 1, v=100.0)
    Uh[15, 2] = 10000.0
    bh[15] = -10000.0

    # Saturating command broadcasts. Suppression at the other two
    # icon types makes a shared click head select the requested icon.
    def command_pair(a, b, own_icon, source_feature=None):
        for y, x in neighbours:
            for out in (a, b):
                rec(out, a, y, x)
                rec(out, b, y, x, -1.0)
        bh[b] = -1.0
        for out in (a, b):
            if source_feature is not None:
                Uh[out, source_feature] = 1.0
            for icon in (14, 15, 16):
                if icon != own_icon:
                    Uh[out, icon] -= 10.0

    command_pair(9, 10, 14, 23)
    command_pair(11, 12, 16)
    command_pair(13, 14, 15, 24)
    for out in (11, 12):
        rec(out, 15)

    Wz, bz = p["Wz"], p["bz"]

    # Directional potentials, with the readout offset removed.
    for j in range(4):
        Wz[j, 5 + j] = 1.0
        bz[j] = -10000.0

    Wz[4, 32 + 17] = 1.0
    Wz[5, 32 + 18] = 1.0
    Wz[6, 15] = 1.0

    # Automatic shrink only when both the candidate and icon exist.
    Wz[7, 32 + 27] = 1.0
    Wz[7, 32 + 14] = 1.0
    bz[7] = -1.0

    Wz[8, 32 + 25] = 1.0
    Wz[8, 32 + 26] = 1.0

    # A firing action is postponed if another legal direction has
    # a better potential than the direction the player currently faces.
    wrong_units = []
    unit = 9
    for facing in range(4):
        for other in range(4):
            if facing == other:
                continue
            Wz[unit, 5 + other] = 1.0
            Wz[unit, 5 + facing] = -1.0
            Wz[unit, 32 + 10 + facing] = 40000.0
            bz[unit] = -40000.01
            wrong_units.append(unit)
            unit += 1

    Wm, Wmz, Wma, bm = (
        p["Wm"], p["Wmz"], p["Wma"], p["bm"]
    )

    # Coordinate history.
    Wmz[0, 4] = 1.0
    Wmz[1, 5] = 1.0
    Wm[2, 0] = 1.0
    Wm[3, 1] = 1.0

    for axis, (zindex, oldindex) in enumerate(((4, 0), (5, 1))):
        for side, sign in enumerate((1.0, -1.0)):
            out = 4 + 2 * axis + side
            Wmz[out, zindex] = 30000.0 * sign
            Wm[out, oldindex] = -30000.0 * sign
            bm[out] = -3.0

    # Saturated pairwise directional rankings.
    comparisons = []
    unit = 8
    for i in range(4):
        for j in range(i + 1, 4):
            Wmz[unit, i] = 100.0
            Wmz[unit, j] = -100.0
            comparisons.append((unit, i, j))
            unit += 1

    # No reachable legal neighbour.
    Wmz[14, :4] = -10.0
    bm[14] = 5.0

    Wmz[15, 6] = 10.0           # firing opportunity
    Wmz[16, 7] = 5.0            # shrink request
    Wmz[17, 8] = 5.0            # grid entry in progress

    for out, zindex in enumerate(wrong_units, 18):
        Wmz[out, zindex] = 100.0

    # Delayed stationary-move test.
    Wm[30, 4:8] = -10.0
    for action in (1, 2, 3, 4):
        Wma[30, action] = 10.0
    bm[30] = -45.0

    Wo, bo = p["Wo"], p["bo"]
    bo[:] = -1.0e6
    bo[1:5] = 0.0

    for unit, i, j in comparisons:
        Wo[1 + i, 128 + unit] += 1.0
        Wo[1 + j, 128 + unit] -= 1.0

    bo[6] = -30.0
    Wo[6, 128 + 14] = 40.0
    Wo[6, 128 + 15] = 70.0
    Wo[6, 128 + 16] = 40.0
    Wo[6, 128 + 17] = 2000.0
    Wo[6, 128 + 18:128 + 30] = -140.0
    Wo[6, 128 + 30] = 35.0

    # Click map: highlighted cells first; otherwise an icon with
    # its matching bounded command flag.
    Wc = p["Wc"][0]
    Wc[26] = 1.0e6
    for icon in (14, 15, 16):
        Wc[icon] = 1000.0
    Wc[32 + 9] = 10.0
    Wc[32 + 10] = -10.0
    Wc[32 + 11] = 30.0
    Wc[32 + 12] = -30.0
    Wc[32 + 13] = 2.0
    Wc[32 + 14] = -2.0

    return p
