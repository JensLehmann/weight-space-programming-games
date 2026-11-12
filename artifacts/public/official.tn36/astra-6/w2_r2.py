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
    colours = (8, 9, 11, 15)
    colour_order = {11: 0, 9: 1, 8: 2, 15: 3}
    nf = 0

    # Local object recognition.
    def pattern(mask, magnitude=1.0, hit=1.0, coloured=False):
        nonlocal nf
        ch = nf
        nf += 1
        n = int(mask.sum())
        for y in range(5):
            for x in range(5):
                inside = bool(mask[y, x])
                for c in colours:
                    w = magnitude if inside else -magnitude
                    if inside and coloured:
                        w += colour_order[c] / float(n)
                    p["W1"][ch, c, y, x] = w
        p["b1"][ch] = hit - magnitude * n
        return ch

    def patch(image, ax, ay):
        out = np.zeros((5, 5), bool)
        for y in range(5):
            for x in range(5):
                ix, iy = ax + x - 2, ay + y - 2
                if 0 <= iy < image.shape[0] and 0 <= ix < image.shape[1]:
                    out[y, x] = image[iy, ix]
        return out

    def rotate_point(x, y, n, turns):
        for _ in range(turns):
            x, y = n - 1 - y, x
        return x, y

    avatar = np.ones((4, 4), bool)
    avatar[3, 1:3] = False
    sources = []
    for turns in range(4):
        mask = np.zeros((5, 5), bool)
        mask[1:, 1:] = np.rot90(avatar, -turns)
        sources.append((pattern(mask), 1, 1, 1))

    large_avatar = np.repeat(np.repeat(avatar, 2, 0), 2, 1)
    base = patch(large_avatar, 2, 6)
    for turns in range(4):
        ax, ay = rotate_point(2, 6, 8, turns)
        sources.append((pattern(np.rot90(base, -turns)), ax, ay, 2))

    goal = np.zeros((6, 6), bool)
    goal[1:5, 0] = True
    goal[1:5, 5] = True
    goal[4, 2:4] = True
    goal[5, :] = True
    targets = []
    for scale, anchor in ((1, (2, 3)), (2, (3, 9))):
        image = np.repeat(np.repeat(goal, scale, 0), scale, 1)
        base = patch(image, *anchor)
        for turns in range(4):
            mask = np.rot90(base, -turns)
            ax, ay = rotate_point(*anchor, 6 * scale, turns)
            a = pattern(mask, 16.0, 10.0, True)
            b = pattern(mask, 16.0, 0.0, True)
            targets.append((a, b, ax, ay, scale))

    # Recognize bits only when surrounded by their black editing panel.
    def marker(vertical):
        nonlocal nf
        ch = nf
        nf += 1
        tested = 0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if vertical:
                    inspect = True
                    line = dx == 0 and -1 <= dy <= 1
                else:
                    inspect = -1 <= dy <= 1 or (dy == 2 and dx != 0)
                    line = dy == 0 and -1 <= dx <= 1
                if inspect:
                    tested += 1
                    if line:
                        p["W1"][ch, 1, dy + 2, dx + 2] = 1
                        p["W1"][ch, 5, dy + 2, dx + 2] = 1
                    else:
                        p["W1"][ch, 0, dy + 2, dx + 2] = 1
        p["b1"][ch] = 1 - tested
        return ch

    markers = (marker(False), marker(True))

    X = nf
    nf += 1
    p["W1"][X, 17, 2, 2] = 1
    Y = nf
    nf += 1
    p["W1"][Y, 16, 2, 2] = 1
    FIVE = nf
    nf += 1
    p["W1"][FIVE, 5, 2, 2] = 1

    WALL = nf
    nf += 1
    p["W1"][WALL, 6, 2, 2] = 1
    p["W1"][WALL, 6, 2, 3] = -1

    RUN_PATCH = nf
    nf += 1
    p["W1"][RUN_PATCH, 9, :, :] = 1
    p["b1"][RUN_PATCH] = -24

    CHECKPOINT = nf
    nf += 1
    for y in range(4):
        for x in range(4):
            p["W1"][CHECKPOINT, 11, y + 1, x + 1] = (
                1 if (x + y) % 2 == 0 else -1
            )
    p["b1"][CHECKPOINT] = -7
    assert nf == 32

    def add2(o, i, w, dy=0, dx=0):
        p["W2"][o, i, dy + 1, dx + 1] += w

    SX, SY, SS = 0, 1, 2
    TX, TY, TS = 4, 5, 6
    COMMON = 8
    SITE, UX, UNX, UY, UNY, ROLE = 10, 11, 12, 13, 14, 15
    ZERO, ONE, RUN, TOP = 16, 17, 18, 19
    SOURCE_FLAG, GOAL_FLAG = 20, 21
    CP_FLAG, CP_X, CP_Y = 22, 23, 24
    WALL_FLAG, WALL_INV_X = 25, 26
    gate = 40.0

    for out, coord in ((SX, X), (SY, Y), (TX, X), (TY, Y)):
        add2(out, coord, 63.0 / 4.0)
        p["b2"][out] = -gate

    for ch, ax, ay, scale in sources:
        add2(SX, ch, gate - ax / 4.0)
        add2(SY, ch, gate - ay / 4.0)
        add2(SS, ch, scale)
        add2(COMMON, ch, 1)
        add2(SOURCE_FLAG, ch, 1)

    for a, b, ax, ay, scale in targets:
        for out, value in (
            (TX, gate + (scale - ax) / 4.0),
            (TY, gate + (scale - ay) / 4.0),
            (TS, scale), (COMMON, 1), (GOAL_FLAG, 1),
        ):
            add2(out, a, value / 10.0)
            add2(out, b, -value / 10.0)

    add2(CP_FLAG, CHECKPOINT, 1)
    for out, coord in ((CP_X, X), (CP_Y, Y)):
        add2(out, coord, 63.0 / 4.0)
        add2(out, CHECKPOINT, gate - 0.25)
        p["b2"][out] = -gate

    add2(WALL_FLAG, WALL, 1)
    add2(WALL_INV_X, WALL, gate)
    add2(WALL_INV_X, X, -63.0 / 4.0)
    p["b2"][WALL_INV_X] = 16.0 - gate

    # Six computational sites surround each editable bit.
    sites = (
        (0, 0, -1), (1, 0, 1),
        (2, -1, -1), (3, -1, 1),
        (4, 1, -1), (5, 1, 1),
    )
    add2(UX, X, 1)
    add2(UNX, X, -1)
    add2(UY, Y, 1)
    add2(UNY, Y, -1)
    for out in (UX, UNX, UY, UNY):
        p["b2"][out] = -gate

    for role, dy, dx in sites:
        for ch in markers:
            oy, ox = -dy, -dx
            add2(SITE, ch, 1, oy, ox)
            add2(ROLE, ch, role, oy, ox)
            add2(UX, ch, gate - dx / 63.0, oy, ox)
            add2(UNX, ch, gate + 1 + dx / 63.0, oy, ox)
            add2(UY, ch, gate - dy / 63.0, oy, ox)
            add2(UNY, ch, gate + 1 + dy / 63.0, oy, ox)

    for ch in markers:
        add2(ZERO, ch, 1)
        add2(ONE, ch, 1)
    add2(ZERO, FIVE, -1)
    add2(ONE, FIVE, 1)
    p["b2"][ONE] = -1
    add2(RUN, RUN_PATCH, 10)
    add2(RUN, Y, 1)
    p["b2"][RUN] = -10.5
    add2(TOP, Y, -63)
    p["b2"][TOP] = 1

    def rm(ch):
        return 32 + ch

    def cm(ch):
        return 64 + ch

    OFFSET = 32.0
    BASE = 20.0

    # Geometry and row-wise candidate information.
    for out, (pos, neg) in enumerate(((SX, TX), (SY, TY), (SS, TS))):
        p["W3"][out, rm(pos)] = 1
        p["W3"][out, rm(neg)] = -1
        p["W3"][out, rm(COMMON)] = OFFSET

    F_COMMON, F_TOP, F_SOURCE = 3, 4, 5
    F_CAND, F_CPX, F_CPY, F_WALL, F_SX = 6, 7, 8, 9, 10
    p["W3"][F_COMMON, rm(COMMON)] = 1
    p["W3"][F_TOP, TOP] = 1
    p["W3"][F_SOURCE, rm(SOURCE_FLAG)] = 1
    p["W3"][F_CAND, rm(CP_FLAG)] = 1
    p["W3"][F_CAND, rm(GOAL_FLAG)] = 1
    p["W3"][F_CPX, rm(CP_X)] = 1
    p["W3"][F_CPY, rm(CP_Y)] = 1
    p["W3"][F_SX, rm(SX)] = 1

    # Leftmost wall's right edge, measured only on the player's row.
    p["W3"][F_WALL, rm(WALL_FLAG)] = 16.25
    p["W3"][F_WALL, rm(WALL_INV_X)] = -1
    p["W3"][F_WALL, rm(SOURCE_FLAG)] = gate
    p["b3"][F_WALL] = -gate

    # Compile intervals for:
    # shrink, initial left, up, down, right, remaining left.
    # Five count inputs are L, R, U, D and remaining-left.
    allow = np.zeros((5, 36), np.float64)
    endpoint_bias = np.full(36, -128.0, np.float64)

    def endpoint(role, bit, terms):
        t = 6 * role + bit
        endpoint_bias[t] = 0
        for term in terms:
            allow[term, t] = 1

    # First interval end/start.
    endpoint(0, 0, (0, 2, 3, 4))
    endpoint(0, 1, (0, 1, 2, 3, 4))
    endpoint(0, 3, ())
    endpoint(0, 5, (0, 2, 4))
    endpoint(1, 1, (0, 2, 4))
    endpoint(1, 5, (0, 4))
    # Second interval supplies the final leftward leg.
    endpoint(2, 0, (0, 1, 2, 3))
    endpoint(3, 0, (0, 1, 2, 3, 4))

    tables = [32.0 * (a - 1.0) for a in allow]
    tables.append(endpoint_bias)

    def spline(v):
        coeff = np.zeros(35, np.float64)
        coeff[0] = v[1] - v[0]
        for j in range(1, 35):
            coeff[j] = v[j + 1] - 2 * v[j] + v[j - 1]
        return float(v[0]), coeff

    splines = [spline(v) for v in tables]
    knots = sorted({
        j for _, c in splines for j in range(35) if c[j] != 0
    })
    assert len(knots) <= 18

    hinge = {}
    f = 11
    for knot in knots:
        hinge[knot] = f
        p["W3"][f, UY] = 21
        p["W3"][f, cm(UNY)] = 21
        p["W3"][f, ROLE] = 6
        p["b3"][f] = -21 - knot
        f += 1

    F_SLOT = f
    f += 1
    p["W3"][F_SLOT, UX] = 63.0 / 5.0
    p["W3"][F_SLOT, rm(UNX)] = 63.0 / 5.0
    p["b3"][F_SLOT] = -63.0 / 5.0

    F_ZERO, F_ONE = f, f + 1
    f += 2
    assert f <= 32
    p["W3"][F_ZERO, ZERO] = 1
    p["W3"][F_ONE, ONE] = 1
    p["W3"][F_ZERO, RUN] = 2
    p["W3"][F_ONE, RUN] = 2

    # Recurrent state:
    # 0,1 displacement; 2 shrink count;
    # 3 upward scan; 4,5 selected-destination corrections;
    # 6 wall/target difference; 7..11 gated instruction counts;
    # 12,13 clipped endpoints; 14,15 bit mismatches.
    for h in range(3):
        p["Wh"][h, h, 0, 1] = 1
        p["Uh"][h, h] = 1
        p["Uh"][h, F_COMMON] = -OFFSET
        p["Uh"][h, F_TOP] = BASE

    # Scan upward from the player, stopping at the first candidate row.
    p["Wh"][3, 3, 2, 1] = 1
    p["Uh"][3, F_SOURCE] = 1
    p["Uh"][3, F_CAND] = -2

    correction_offset = 32.0
    selection_gate = 256.0

    for h, field, cp in ((4, 0, F_CPX), (5, 1, F_CPY)):
        p["Wh"][h, field, 0, 1] = 1
        p["Wh"][h, 3, 2, 1] = selection_gate
        p["Uh"][h, F_SOURCE] = selection_gate
        p["Uh"][h, F_CAND] = selection_gate
        p["Uh"][h, cp] = 1
        p["bh"][h] = (
            correction_offset - BASE - 2 * selection_gate
        )

    # The scan's stopped-wave difference is a unit selection flag.
    for field, correction in ((0, 4), (1, 5)):
        p["Wh"][field, correction, 1, 1] = -1
        p["Wh"][field, 3, 2, 1] += correction_offset
        p["Wh"][field, 3, 1, 1] -= correction_offset
        p["Uh"][field, F_SOURCE] += correction_offset

    # Broadcast wall-right minus selected-target-x.
    p["Wh"][6, 6, 0, 1] = 1
    p["Uh"][6, 0] = 1
    p["Uh"][6, F_COMMON] = -OFFSET
    p["Uh"][6, F_SX] = -1
    p["Uh"][6, F_WALL] = 1
    p["Uh"][6, F_TOP] = BASE
    p["Wh"][6, 4, 1, 1] = -1
    p["Wh"][6, 3, 2, 1] = correction_offset
    p["Wh"][6, 3, 1, 1] = -correction_offset
    p["Uh"][6, F_SOURCE] = correction_offset

    def add_spline(h, data):
        intercept, coeff = data
        p["bh"][h] += intercept
        for knot, ch in hinge.items():
            p["Uh"][h, ch] += coeff[knot]

    count_inputs = (
        (0, 1, -BASE),       # left
        (0, -1, BASE),       # right
        (1, 1, -BASE),       # up
        (1, -1, BASE),       # down
        (6, 1, -BASE),       # postponed left
    )
    for i, (field, sign, bias) in enumerate(count_inputs):
        h = 7 + i
        p["Wh"][h, field, 1, 1] = sign
        p["bh"][h] = bias
        add_spline(h, splines[i])

    for h, threshold in ((12, 0.5), (13, 1.5)):
        p["Wh"][h, 2, 1, 1] = 2
        for count in range(7, 11):
            p["Wh"][h, count, 1, 1] = 2
        p["Wh"][h, 11, 1, 1] = -2
        p["Uh"][h, F_SLOT] = -2
        p["bh"][h] = -2 * BASE - threshold
        add_spline(h, splines[5])

    terms = (
        (0, -1, 1), (0, 1, -1),
        (-1, -1, 1), (-1, 1, -1),
        (1, -1, 1), (1, 1, -1),
    )
    for dy, dx, sign in terms:
        p["Wh"][14, 12, dy + 1, dx + 1] += sign
        p["Wh"][14, 13, dy + 1, dx + 1] -= sign
        p["Wh"][15, 12, dy + 1, dx + 1] -= sign
        p["Wh"][15, 13, dy + 1, dx + 1] += sign

    p["Uh"][14, F_ZERO] = 64
    p["bh"][14] = -64
    p["Uh"][15, F_ONE] = 64
    p["bh"][15] = -63

    p["Wc"][0, F_ZERO] = 1
    p["Wc"][0, F_ONE] = 1
    p["Wc"][0, 32 + 14] = 4
    p["Wc"][0, 32 + 15] = 4
    p["bo"][6] = 1
    return p
