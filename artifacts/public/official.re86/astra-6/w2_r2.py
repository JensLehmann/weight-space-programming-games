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

    colors = (8, 9, 11)
    cardinal = ((-1, 0), (1, 0), (0, -1), (0, 1))
    diagonal = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    surrounding = cardinal + diagonal
    outer = tuple(
        (dy, dx)
        for dy in range(-2, 3)
        for dx in range(-2, 3)
        if max(abs(dy), abs(dx)) == 2
    )

    def c1(o, i, dy=0, dx=0, v=1.0):
        p["W1"][o, i, dy + 2, dx + 2] += v

    def c2(o, i, dy=0, dx=0, v=1.0):
        p["W2"][o, i, dy + 1, dx + 1] += v

    def ch(o, i, dy=0, dx=0, v=1.0):
        p["Wh"][o, i, dy + 1, dx + 1] += v

    # First layer: paired ReLU ramps form binary detections.
    target = {}
    for j, color in enumerate(colors):
        a, b = 2 * j, 2 * j + 1
        target[color] = (a, b)
        for out, bias in ((a, -18), (b, -19)):
            c1(out, color, v=16)
            for dy, dx in surrounding:
                c1(out, 4, dy, dx)
            for dy, dx in outer:
                c1(out, 4, dy, dx, -8)
            p["b1"][out] = bias

    # Colour-independent marker centres, including covered centres.
    for out, bias in ((6, -18), (7, -19)):
        for color in (0, 6, 7, 8, 9, 10, 11, 12, 13, 14):
            c1(out, color, v=16)
        for dy, dx in surrounding:
            c1(out, 4, dy, dx)
        for dy, dx in outer:
            c1(out, 4, dy, dx, -8)
        p["b1"][out] = bias

    # Paint interiors: every immediate neighbour must be either
    # the same colour or border colour 2, with an outer border present.
    paint = {}
    for j, color in enumerate(colors):
        a, b = 8 + 2 * j, 9 + 2 * j
        paint[color] = (a, b)
        for out, bias in ((a, -512), (b, -513)):
            c1(out, color, v=256)
            for dy, dx in surrounding:
                c1(out, color, dy, dx, 32)
                c1(out, 2, dy, dx, 32)
            for dy, dx in outer:
                c1(out, 2, dy, dx)
            p["b1"][out] = bias

    # Local colour evidence at the zero selection marker.
    local_color = {}
    for j, color in enumerate(colors):
        a, b = 14 + 2 * j, 15 + 2 * j
        local_color[color] = (a, b)
        for out, bias in ((a, -12), (b, -13)):
            c1(out, 0, v=10)
            for dy, dx in surrounding:
                c1(out, color, dy, dx)
            p["b1"][out] = bias

    ZERO, SHAPE, ROW, COL = 20, 21, 22, 23
    c1(ZERO, 0)
    for color in range(6, 15):
        c1(SHAPE, color)
    c1(ROW, 16)
    c1(COL, 17)

    # Thin shape pixels, excluding marker rings and paint pots.
    thin_color = {}
    for j, color in enumerate(colors):
        a, b = 24 + 2 * j, 25 + 2 * j
        thin_color[color] = (a, b)
        for out, bias in ((a, -33), (b, -34)):
            c1(out, color, v=32)
            for dy, dx in surrounding:
                c1(out, 5, dy, dx)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    c1(out, 2, dy, dx, -8)
                    c1(out, 4, dy, dx, -8)
            p["b1"][out] = bias

    # Second layer: target-coordinate ramps.
    for j, color in enumerate(colors):
        a, b = target[color]
        for k, coord, sign, offset in (
            (0, COL, 1, 1),
            (1, COL, -1, 2),
            (2, ROW, 1, 1),
            (3, ROW, -1, 2),
        ):
            out = 4 * j + k
            c2(out, a, v=10)
            c2(out, b, v=-10)
            c2(out, coord, v=sign)
            p["b2"][out] = offset - 10

    for out, coord in ((12, COL), (13, ROW)):
        c2(out, ZERO, v=10)
        c2(out, coord)
        p["b2"][out] = -9
    c2(14, ZERO)

    # Selected local geometry: plus, X, hollow, horizontal.
    for out, offsets in ((15, cardinal), (16, diagonal)):
        c2(out, ZERO, v=10)
        for dy, dx in offsets:
            c2(out, SHAPE, dy, dx, 0.5)
        p["b2"][out] = -11

    c2(17, ZERO, v=10)
    for dy, dx in surrounding:
        c2(17, SHAPE, dy, dx, -1)
    p["b2"][17] = -9

    c2(18, ZERO, v=10)
    c2(18, SHAPE, 0, -1)
    c2(18, SHAPE, 0, 1)
    c2(18, SHAPE, -1, 0, -1)
    c2(18, SHAPE, 1, 0, -1)
    p["b2"][18] = -11

    for j, color in enumerate(colors):
        a, b = local_color[color]
        c2(19 + j, a, v=2)
        c2(19 + j, b, v=-2)
        a, b = thin_color[color]
        c2(19 + j, a)
        c2(19 + j, b, v=-1)

        a, b = paint[color]
        for out, coord in ((22 + 2 * j, COL),
                           (23 + 2 * j, ROW)):
            c2(out, a, v=10)
            c2(out, b, v=-10)
            c2(out, coord)
            p["b2"][out] = -9

    c2(28, 6)
    c2(28, 7, v=-1)
    c2(29, ROW)
    c2(30, COL)

    # Third layer: rotated colour-8 bounds for diamond placement.
    for out, aa, bb in (
        (0, 0, 2), (1, 1, 3),
        (2, 0, 3), (3, 1, 2),
    ):
        p["W3"][out, aa] = 0.5
        p["W3"][out, bb] = 0.5

    eps = 0.001

    def axial_base(out, j, axis):
        a, b, c, d = range(4 * j, 4 * j + 4)
        p["W3"][out, a] += 10.0 / 3.0
        p["W3"][out, b] += 10.0 / 3.0
        if axis == "x":
            p["W3"][out, 64 + c] += 1
            p["W3"][out, 64 + d] += 1
            p["W3"][out, 32 + a] -= 1
            p["W3"][out, 32 + b] -= 1
        else:
            p["W3"][out, 32 + a] += 1
            p["W3"][out, 32 + b] += 1
            p["W3"][out, 64 + c] -= 1
            p["W3"][out, 64 + d] -= 1
        p["b3"][out] = -9

    for j in (1, 2):
        for off, axis in ((0, "x"), (1, "x"),
                          (2, "y"), (3, "y")):
            axial_base(4 * j + off, j, axis)
        p["W3"][4 * j + 1, 30] += eps
        p["W3"][4 * j + 3, 29] += eps

    # Widest horizontal colour-8 target row.
    for out in (12, 13, 14):
        p["W3"][out, 0] = 10.0 / 3.0
        p["W3"][out, 1] = 10.0 / 3.0
        p["W3"][out, 32] = 1
        p["W3"][out, 33] = 1
        p["b3"][out] = -12
    p["W3"][13, 32] += eps / 2
    p["W3"][13, 33] -= eps / 2
    p["b3"][13] += eps / 2
    p["W3"][14, 29] += eps

    # An upper horizontal pair gives an X centre below the pair.
    for out in (15, 16, 17):
        p["W3"][out, 4] = 10.0 / 3.0
        p["W3"][out, 5] = 10.0 / 3.0
        p["W3"][out, 36] = 4
        p["W3"][out, 37] = 4
        p["W3"][out, 29] = -1
        p["b3"][out] = -22.1
    p["W3"][16, 36] += eps / 2
    p["W3"][16, 37] -= eps / 2
    p["b3"][16] += eps / 2
    p["W3"][17, 29] += eps
    p["W3"][17, 36] += eps / 2
    p["W3"][17, 37] += eps / 2
    p["b3"][17] -= 1.5 * eps

    # Pack geometry into the integer part of active coordinates.
    for out, coord in ((18, 12), (19, 13)):
        p["W3"][out, coord] = 1
        for inp, weight in ((15, 1), (16, 2), (17, 4), (18, 8)):
            p["W3"][out, inp] += 10 * weight

    # A local match scores 6; a hollow axial match scores 2.
    for j in range(3):
        out, inp = 20 + j, 19 + j
        p["W3"][out, inp] = 1
        p["W3"][out, 32 + inp] = 1
        p["W3"][out, 64 + inp] = 1
        p["W3"][out, 14] = 10
        p["b3"][out] = -10

    for j in range(6):
        p["W3"][23 + j, 22 + j] = 1
    p["W3"][29, 28] = 16
    p["W3"][30, 29] = 1
    p["W3"][31, 30] = 1

    # Recurrent Hough transform:
    # 0 clock; 1..3 delayed source; 4..7 Manhattan-shell recurrence;
    # 8..9 diamond votes; 10..13 diagonal rays; 14..15 X votes.
    launch, clock_scale = 114, 64.0
    ex, ey = 0.25, 0.002

    ch(0, 0)
    p["bh"][0] = 1
    for out in (1, 2):
        ch(out, 0, v=clock_scale)
        p["bh"][out] = -clock_scale * launch
    p["Uh"][1, 29] = 1
    ch(3, 1)
    ch(3, 2, v=-1)

    for dy, dx in cardinal:
        ch(4, 4, dy, dx)
        ch(4, 6, dy, dx)
    ch(4, 5, v=-2)
    for dy, dx in diagonal:
        ch(4, 5, dy, dx, -1)
    ch(4, 7, v=-1)
    ch(4, 1)
    ch(4, 2, v=-1)
    ch(4, 3, v=-1)
    ch(5, 4)
    ch(6, 5)
    ch(7, 6)

    for out in (8, 9):
        for dy, dx in cardinal:
            ch(out, 4, dy, dx)
            ch(out, 6, dy, dx)
        ch(out, 5, v=-4)
        for dy, dx in diagonal:
            ch(out, 5, dy, dx, -1)
        p["Uh"][out, 31] = -ex
        p["bh"][out] = ex
    p["Uh"][9, 30] = ey

    for out, (dy, dx) in zip(range(10, 14), diagonal):
        ch(out, out, dy, dx)
        ch(out, 1)
        ch(out, 2, v=-1)

    for out in (14, 15):
        for ray in range(10, 14):
            ch(out, ray)
        p["Uh"][out, 29] = -3
        p["Uh"][out, 30] = -ex
        p["bh"][out] = ex
    p["Uh"][15, 31] = ey

    # Readout construction.
    def gv(index, value=1.0):
        a = np.zeros(96, dtype=np.float64)
        a[index] = value
        return a

    def fg(index, value=1.0):
        return gv(32 + index, value)

    nz = 0

    def Z(weights=None, bias=0.0):
        nonlocal nz
        if weights is None:
            weights = np.zeros(96)
        if nz >= 128:
            raise RuntimeError("readout capacity exceeded")
        p["Wz"][nz] = weights
        p["bz"][nz] = bias
        out = np.zeros(128, dtype=np.float64)
        out[nz] = 1
        nz += 1
        return out

    def clip(weights, bias=0.0):
        return Z(weights, bias) - Z(weights, bias - 1)

    one = Z(bias=1)

    def geometry_indicator(code):
        lo, hi = 10 * code + 1, 10 * code + 2
        s = fg(18)
        return (
            Z(s, 1 - lo) - Z(s, -lo)
            - Z(s, -hi) + Z(s, -hi - 1)
        )

    plus_type = geometry_indicator(1)
    x_type = geometry_indicator(2)
    hollow_type = geometry_indicator(4)
    line_type = geometry_indicator(8)
    type_value = (
        plus_type + 2 * x_type + 4 * hollow_type + 8 * line_type
    )
    active_present = clip(10 * fg(18))
    active_x = Z(fg(18)) - 10 * type_value - active_present
    active_y = Z(fg(19)) - 10 * type_value - active_present

    current = {
        color: clip(fg(20 + j), -5)
        for j, color in enumerate(colors)
    }
    hollow_eight = clip(fg(20), -1)

    plus_x, plus_y = {}, {}
    valid, current_valid = {}, {}
    for j, color in ((1, 9), (2, 11)):
        plus_x[color] = Z(
            (fg(4 * j + 1) - fg(4 * j)) / eps
        )
        plus_y[color] = Z(
            (fg(4 * j + 3) - fg(4 * j + 2)) / eps
        )
        evidence = 64 * (fg(4 * j) + fg(4 * j + 2))
        valid[color] = clip(evidence, -64 * 2.005)
        current_valid[color] = clip(
            evidence + 1024 * fg(20 + j),
            -64 * 2.005 - 6144,
        )
    stay = current_valid[9] + current_valid[11]

    paint_x, paint_y, paint_present = {}, {}, {}
    for j, color in enumerate(colors):
        raw_x = Z(fg(23 + 2 * j))
        paint_x[color] = Z(fg(23 + 2 * j), -1)
        paint_y[color] = Z(fg(24 + 2 * j), -1)
        paint_present[color] = raw_x - paint_x[color]

    u_hi, u_lo, v_hi, v_lo = [Z(fg(i)) for i in range(4)]
    compact_x = 0.5 * (u_hi - u_lo + v_hi - v_lo + one)
    compact_y = 0.5 * (u_hi - u_lo - v_hi + v_lo + one)
    present8 = clip(10 * fg(0))
    large = clip(
        100 * (fg(0) + fg(1) + fg(2) + fg(3)),
        -631,
    )
    small = present8 - large
    u_valid = clip(100 * (fg(0) + fg(1)), -304.5)
    v_valid = clip(100 * (fg(2) + fg(3)), -304.5)

    line_x = Z((fg(13) - fg(12)) / eps)
    line_y = Z((fg(14) - fg(12)) / eps)
    pair_x = Z((fg(16) - fg(15)) / eps)
    pair_y = Z((fg(17) - fg(15)) / eps)
    pair_present = clip(100 * fg(15))

    def decode_hough(main, secondary):
        score = Z(gv(main))
        floor = np.zeros(128, dtype=np.float64)
        for n in range(1, 11):
            low = 16 * (n - 1) + ex
            high = 16 * n
            floor += (
                Z(gv(main), -low) - Z(gv(main), -high)
            ) * (16 / (high - low))
        primary_coordinate = (score - floor) / ex
        secondary_coordinate = Z(
            (gv(secondary) - gv(main)) / ey
        )
        return primary_coordinate, secondary_coordinate

    diamond_left, diamond_y = decode_hough(8, 9)
    diamond_x = one - diamond_left
    cross_up, cross_x = decode_hough(14, 15)
    cross_y = one - cross_up
    assert nz <= 128

    # Tanh policy units implement conjunctions of geometry, colour,
    # target compatibility and signed coordinate error.
    p["bo"][:] = -1000
    p["bo"][1:5] = 0
    p["bo"][5] = 0.55

    nm = 0
    movement_gain = 100.0
    gate_gain = 256.0
    margin = 1.75 / 63.0

    def direction(action, error, gate, gate_bias, scale):
        nonlocal nm
        if nm >= 64:
            raise RuntimeError("policy capacity exceeded")
        p["Wmz"][nm] = movement_gain * error + gate_gain * gate
        p["bm"][nm] = (
            -movement_gain * margin + gate_gain * gate_bias
        )
        p["Wo"][action, 128 + nm] = scale / 2
        p["bo"][action] += scale / 2
        nm += 1

    def navigate(dx, dy, gate, gate_bias, painting=False):
        vertical = 2.1 if painting else 1.0
        horizontal = 2.0 if painting else 1.05
        direction(1, -dy, gate, gate_bias, vertical)
        direction(2, dy, gate, gate_bias, vertical)
        direction(3, -dx, gate, gate_bias, horizontal)
        direction(4, dx, gate, gate_bias, horizontal)

    def fetch(color, gate, bias, hollow=False):
        colour_match = hollow_eight if hollow else current[color]
        navigate(
            paint_x[color] - active_x,
            paint_y[color] - active_y,
            gate + paint_present[color] - colour_match,
            bias - 1,
            painting=True,
        )

    navigate(
        pair_x - active_x, pair_y - active_y,
        x_type + pair_present + current[9], -3,
    )
    navigate(
        cross_x - active_x, cross_y - active_y,
        x_type - pair_present + current[8], -2,
    )
    navigate(
        compact_x - active_x, compact_y - active_y,
        hollow_type + small + u_valid + v_valid + hollow_eight,
        -5,
    )
    navigate(
        diamond_x - active_x, diamond_y - active_y,
        hollow_type + large + hollow_eight, -3,
    )
    navigate(
        line_x - active_x, line_y - active_y,
        line_type + current[8], -2,
    )
    for color in (9, 11):
        navigate(
            plus_x[color] - active_x,
            plus_y[color] - active_y,
            plus_type + valid[color] + current[color], -3,
        )

    fetch(9, x_type + pair_present, -2)
    fetch(8, x_type - pair_present, -1)
    fetch(
        8, hollow_type + small + u_valid + v_valid, -4,
        hollow=True,
    )
    fetch(8, hollow_type + large, -2, hollow=True)
    fetch(8, line_type, -1)
    fetch(9, plus_type + valid[9] - stay, -2)
    fetch(11, plus_type + valid[11] - valid[9] - stay, -2)

    assert nm == 56

    # Suppress immediate reversals. This allows another useful axis
    # to take precedence when overlapping pieces perturb classification.
    opposite = {1: 2, 2: 1, 3: 4, 4: 3}
    for previous in range(1, 5):
        p["Wma"][nm, previous] = 10
        p["Wo"][opposite[previous], 128 + nm] = -1.1
        nm += 1

    # Continue a movement if the zero marker is temporarily occluded.
    for action in range(1, 5):
        p["Wma"][nm, action] = 20
        p["Wmz"][nm] = -20 * active_present
        p["bm"][nm] = -10
        p["Wo"][action, 128 + nm] = 1.25
        p["bo"][action] += 1.25
        nm += 1

    p["bo"][1] += 0.0004
    p["bo"][2] += 0.0003
    p["bo"][3] += 0.0002
    p["bo"][4] += 0.0001

    assert nm == 64
    return p
