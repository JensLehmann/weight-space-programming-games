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

    W1, b1 = p["W1"], p["b1"]
    W2, b2 = p["W2"], p["b2"]
    W3, b3 = p["W3"], p["b3"]
    cross = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def k1(o, i, dy=0, dx=0, v=1.0):
        W1[o, i, dy + 2, dx + 2] += v

    def k2(o, i, dy=0, dx=0, v=1.0):
        W2[o, i, dy + 1, dx + 1] += v

    # Binary body masks, including interface pinholes.
    for first, colour in ((0, 5), (2, 4)):
        for out, bias in ((first, -3.0), (first + 1, -4.0)):
            k1(out, colour, v=3.0)
            for dy, dx in cross:
                k1(out, colour, dy, dx)
            b1[out] = bias

    k1(4, 11)

    # Covered target centres: expand these, but not uncovered targets.
    k1(5, 11, v=4.0)
    for dy, dx in cross:
        k1(5, 5, dy, dx)
        k1(5, 4, dy, dx)
    b1[5] = -7.0

    # Clipped expanded selection markers.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            k1(6, 0, dy, dx)
    W1[7] = W1[6]
    b1[7] = -1.0

    k1(8, 16)
    k1(9, 17)
    k1(10, 16, v=63.0)
    b1[10] = -62.0
    k1(11, 17, v=63.0)
    b1[11] = -62.0

    # Selected-axis detectors.
    for out, far in ((12, (0, 2)), (13, (2, 0))):
        k1(out, 0, v=4.0)
        for dy, dx in cross:
            k1(out, 10, dy, dx)
            k1(out, 0, dy, dx)
        k1(out, 10, far[0], far[1])
        b1[out] = -8.0

    # Cell-spaced 2x2 squares.
    square = ((-1, -1), (-1, 2), (2, -1), (2, 2))
    for dy, dx in square:
        k1(14, 5, dy, dx)
        k1(14, 0, dy, dx)
        k1(15, 11, dy, dx)
    b1[14] = b1[15] = -3.0

    # Cell-spaced corner inputs.
    for colour in (5, 0):
        k1(16, colour)
    for j, (dy, dx) in enumerate(cross):
        for colour in (5, 0):
            k1(17 + j, colour, 2 * dy, 2 * dx)
        k1(21 + j, 11, 2 * dy, 2 * dx)

    # Presence of a vertical reflection axis, selected or otherwise.
    k1(25, 10, -2, 0)
    k1(25, 10, 2, 0)
    for colour in (0, 9, 10):
        k1(25, colour)
    b1[25] = -2.0

    # F2: normalized occupancy and reconstructed targets.
    for out, first in ((0, 0), (1, 2)):
        k2(out, first)
        k2(out, first + 1, v=-1.0)
        k2(out, 10, v=-8.0)
        k2(out, 11, v=-8.0)

    def target(out):
        k2(out, 4)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                k2(out, 5, dy, dx)
        k2(out, 5, v=-1.0)
        k2(out, 10, v=-8.0)
        k2(out, 11, v=-8.0)

    target(2)
    k2(3, 6)
    k2(3, 7, v=-1.0)
    k2(3, 10, v=-8.0)
    k2(3, 11, v=-8.0)

    k2(4, 8)
    k2(5, 9)
    k2(6, 12)
    k2(7, 13)
    k2(8, 12)
    k2(8, 8)
    b2[8] = -1.0
    k2(9, 13)
    k2(9, 9)
    b2[9] = -1.0

    for out, coordinate, sign in (
        (10, 8, 1.0), (11, 9, 1.0),
        (12, 8, -1.0), (13, 9, -1.0),
    ):
        target(out)
        k2(out, coordinate, v=sign)
        b2[out] = -1.0 if sign > 0 else 0.0

    k2(14, 14)
    k2(15, 15)

    # Up-right and up-left source corners; up-right target corners.
    for out, centre, neighbours, side in (
        (16, 16, 17, 3),
        (17, 16, 17, 2),
        (18, 4, 21, 3),
    ):
        k2(out, centre)
        for j, (dy, dx) in enumerate(cross):
            k2(out, neighbours + j, dy, dx,
               1.0 if j in (0, side) else -1.0)
        b2[out] = -2.0

    k2(19, 25)
    k2(20, 10)
    k2(21, 11)

    def local(o, i, v=1.0):
        W3[o, i] += v

    def row(o, i, v=1.0):
        W3[o, 32 + i] += v

    def col(o, i, v=1.0):
        W3[o, 64 + i] += v

    def exclude_padding(o):
        local(o, 20, -8.0)
        local(o, 21, -8.0)

    for out, source in {
        0: 0, 1: 1, 2: 2, 3: 4, 4: 5,
        5: 6, 6: 7, 7: 8, 8: 9,
        9: 10, 10: 11, 11: 12, 12: 13,
        29: 15, 30: 18, 31: 19,
    }.items():
        local(out, source)
        if out not in (3, 4):
            exclude_padding(out)

    def selected_feature(out, feature, coordinate=None,
                         x_weight=0.0):
        local(out, feature, 1.0 if coordinate is None else 2.0)
        local(out, 0, 4.0)
        row(out, 3, 8.0)
        col(out, 3, 8.0)
        b3[out] = -20.0 if coordinate is None else -21.0
        if coordinate is not None:
            local(out, coordinate)
        if x_weight:
            local(out, 5, x_weight)
        exclude_padding(out)

    # Bottommost square row, with tie-breaking x coordinates.
    selected_feature(13, 14)
    selected_feature(14, 14, 4, 0.01)
    selected_feature(15, 14, 4, -0.01)
    for out, x_weight in ((16, 0.01), (17, -0.01)):
        local(out, 15, 2.0)
        local(out, 4)
        local(out, 5, x_weight)
        b3[out] = -1.0
        exclude_padding(out)

    selected_feature(18, 16)
    selected_feature(19, 17)
    selected_feature(20, 16, 4)
    selected_feature(21, 16, 5)

    for out, coordinate in ((22, 4), (23, 5)):
        local(out, 18, 2.0)
        local(out, coordinate)
        b3[out] = -1.0
        exclude_padding(out)

    # Bounding coordinates of the selected original body.
    for out, coordinate, sign in (
        (24, 4, 1.0), (25, 5, 1.0),
        (26, 4, -1.0), (27, 5, -1.0),
    ):
        local(out, 0, 2.0)
        row(out, 3, 8.0)
        col(out, 3, 8.0)
        local(out, coordinate, sign)
        b3[out] = -17.0 if sign > 0 else -16.0
        exclude_padding(out)

    local(28, 0)
    row(28, 3, 8.0)
    col(28, 3, 8.0)
    b3[28] = -16.0
    exclude_padding(28)

    # Recurrent distance transform.
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]
    for dy, dx in cross:
        for out in (0, 1):
            Wh[out, 0, dy + 1, dx + 1] = 1.0
            Wh[out, 1, dy + 1, dx + 1] = -1.0
    for out in (0, 1):
        Uh[out, 0] = 1.0
        Uh[out, 1] = 1.0
    bh[1] = -1.0

    Wh[2, 2, 1, 1] = 1.0
    Wh[2, 0, 1, 1] = 1.0
    Wh[2, 1, 1, 1] = -1.0
    Uh[2, 2] = 1.0
    bh[2] = -1.0

    # Coordinates of the farthest target pixels.
    for out in range(3, 8):
        Wh[out, 2, 1, 1] = -1.0
        Uh[out, 2] = 127.0
        bh[out] = -0.5
    Uh[4, 3] = 0.1
    Uh[5, 3] = -0.1
    Uh[6, 4] = 0.1
    Uh[7, 4] = -0.1

    # Additional small penalty for target pixels not currently covered.
    Uh[8, 2] = 1.0
    Uh[8, 0] = -1.0
    Uh[8, 1] = -1.0

    nz = 0

    def unit(weights=None, bias=0.0):
        nonlocal nz
        i = nz
        nz += 1
        if weights:
            for j, v in weights.items():
                p["Wz"][i, j] += v
        p["bz"][i] = bias
        return i

    def add(*parts):
        result = {}
        for weights, scale in parts:
            for j, v in weights.items():
                result[j] = result.get(j, 0.0) + scale * v
        return result

    def mx(c):
        return 32 + c

    def mean(c):
        return 64 + c

    cost = unit({mean(2): 127.0, 18: -1.0, 24: 1.5})

    def active(raw, bias=0.0):
        return unit(raw, bias), unit(raw, bias - 1.0)

    def gated_error(raw, raw_bias, error, error_bias=0.0):
        return (
            unit(add((raw, 8.0), (error, 1.0)),
                 8.0 * raw_bias - 4.0 + error_bias),
            unit(add((raw, 8.0)), 8.0 * raw_bias - 4.0),
        )

    hraw = {mean(5): 4096.0, mean(6): -4096.0}
    vraw = {j: -v for j, v in hraw.items()}
    ah = active(hraw, -2.0)
    av = active(vraw, -2.0)

    eh = gated_error(
        hraw, -2.0,
        {mx(9): 0.5, mx(11): -0.5, mx(7): -1.0}, 0.5
    )
    ev = gated_error(
        vraw, -2.0,
        {mx(10): 0.5, mx(12): -0.5, mx(8): -1.0}, 0.5
    )

    axis_rejection = {
        mean(5): -16384.0,
        mean(6): -16384.0,
    }

    draw = add(
        ({mx(13): 1.0, mx(29): 1.0}, 1.0),
        (axis_rejection, 1.0),
    )
    ad = active(draw, -1.0)
    dense_y = {
        mx(16): 0.5, mx(17): 0.5,
        mx(14): -0.5, mx(15): -0.5,
    }
    dense_x = {
        mx(16): 50.0, mx(17): -50.0,
        mx(14): -50.0, mx(15): 50.0,
    }
    edy = gated_error(draw, -1.0, dense_y)
    edx = gated_error(draw, -1.0, dense_x)

    craw = add(
        ({
            mx(18): 1.0, mx(30): 1.0,
            mx(19): -1.0, mx(13): -1.0, mx(31): -1.0,
        }, 1.0),
        (axis_rejection, 1.0),
    )
    ac = active(craw, -1.0)
    ecy = gated_error(
        craw, -1.0, {mx(22): 1.0, mx(20): -1.0}
    )
    ecx = gated_error(
        craw, -1.0, {mx(23): 1.0, mx(21): -1.0}
    )

    far_y = (
        unit({4: 5.0, 5: -5.0, mx(24): -0.5, mx(26): 0.5},
             1.5),
        unit(bias=2.0),
    )
    far_x = (
        unit({6: 5.0, 7: -5.0, mx(25): -0.5, mx(27): 0.5},
             1.5),
        unit(bias=2.0),
    )

    edge = {
        1: unit({mx(26): 1.0}, -1.99),
        2: unit({mx(24): 1.0}, -1.975),
        3: unit({mx(27): 1.0}, -1.99),
        4: unit({mx(25): 1.0}, -1.975),
    }

    Wo, bo = p["Wo"], p["bo"]
    bo[:] = -1.0e6
    bo[1:6] = 0.0
    bo[7] = 0.0

    def out_pair(action, pair, amount):
        Wo[action, pair[0]] += amount
        Wo[action, pair[1]] -= amount

    out_pair(1, far_y, -0.2)
    out_pair(2, far_y, 0.2)
    out_pair(3, far_x, -0.2)
    out_pair(4, far_x, 0.2)

    scale = 1000.0
    tolerance = 1.25 / 63.0

    out_pair(1, eh, -scale)
    out_pair(2, eh, scale)
    out_pair(3, ev, -scale)
    out_pair(4, ev, scale)
    for action in (1, 2):
        out_pair(action, ah, -scale * tolerance)
        out_pair(action, av, -1000.0)
    for action in (3, 4):
        out_pair(action, av, -scale * tolerance)
        out_pair(action, ah, -1000.0)

    for presence, ey, ex in ((ad, edy, edx), (ac, ecy, ecx)):
        out_pair(1, ey, -scale)
        out_pair(2, ey, scale)
        out_pair(3, ex, -scale)
        out_pair(4, ex, scale)
        for action in range(1, 5):
            out_pair(action, presence, -scale * tolerance)

    for action in range(1, 5):
        Wo[action, edge[action]] = -20000.0

    Wm, Wmz, Wma, bm = (
        p["Wm"], p["Wmz"], p["Wma"], p["bm"]
    )

    def mem_pair(i, pair, amount):
        Wmz[i, pair[0]] += amount
        Wmz[i, pair[1]] -= amount

    # Candidate direction: continue a move, or turn clockwise after undo.
    clockwise = {1: 4, 4: 2, 2: 3, 3: 1}
    predecessor = {v: k for k, v in clockwise.items()}
    for action in range(1, 5):
        i = action - 1
        Wma[i, action] = 40.0
        Wma[i, 7] = 20.0
        Wm[i, predecessor[action] - 1] = 10.0
        bm[i] = -20.0
        Wo[action, 128 + i] = 0.5
        bo[action] += 0.5

    # Store a small, nearly linear representation of the current cost.
    Wmz[4, cost] = 0.001

    # Previous observation used direct geometric guidance.
    for pair in (ah, av, ad, ac):
        mem_pair(6, pair, 20.0)
    bm[6] = -10.0

    # Reject non-improving ordinary moves.
    Wmz[5, cost] = 10000.0
    Wm[5, 4] = -10000000.0
    Wma[5, 1:5] = 100000.0
    Wm[5, 6] = -50000.0
    bm[5] = 3.0 - 150000.0

    Wo[7, 128 + 5] = 10.0
    for pair in (ah, av, ad, ac):
        out_pair(7, pair, -1000.0)

    # Seven-step undo history.
    Wma[8, 7] = 20.0
    bm[8] = -10.0
    for i in range(9, 15):
        Wm[i, i - 1] = 20.0
        bm[i] = -10.0

    # Eight consecutive moves invite a selection change.
    Wma[16, 1:5] = 20.0
    bm[16] = -10.0
    for i in range(17, 24):
        Wm[i, i - 1] = 20.0
        Wma[i, 1:5] = 20.0
        bm[i] = -30.0
    Wo[5, 128 + 23] = 2.0
    bo[5] += 2.0

    # After four rejected directions, permit three escape moves,
    # then change selection. These states also suppress rejection.
    Wm[32, 8:15] = 5.0
    Wma[32, 7] = 1000.0
    bm[32] = -990.0
    for i in range(33, 36):
        Wm[i, i - 1] = 20.0
        bm[i] = -10.0

    for i in range(32, 35):
        Wm[5, i] = -50000.0
        bm[5] -= 50000.0
        Wo[7, 128 + i] = -50.0
        bo[7] -= 50.0

    Wo[5, 128 + 35] = 50.0
    bo[5] += 50.0

    return p
