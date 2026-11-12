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

    def c1(o, i, v, dy=0, dx=0):
        p["W1"][o, i, dy + 2, dx + 2] += v

    def c2(o, i, v, dy=0, dx=0):
        p["W2"][o, i, dy + 1, dx + 1] += v

    def ch(o, i, v, dy=0, dx=0):
        p["Wh"][o, i, dy + 1, dx + 1] += v

    def row(i):
        return 32 + i

    def col(i):
        return 64 + i

    def merge(*parts):
        result = {}
        for part in parts:
            for i, v in part.items():
                result[i] = result.get(i, 0.0) + v
        return result

    def feature(o, terms, bias=0):
        for i, v in terms.items():
            p["W3"][o, i] += v
        p["b3"][o] = bias

    # Occupancies, coordinates, and orientation evidence.
    for color in (8, 9):
        c1(0, color, 1)
    c1(1, 9, 1)
    for color in (11, 13):
        c1(2, color, 1)
    c1(3, 6, 1)
    c1(4, 16, 1)
    c1(5, 17, 1)
    c1(26, 15, 1)
    c1(27, 12, 1)
    c1(28, 1, 1)
    c1(29, 4, 1)

    # Detect the horizontal image boundary.
    c1(31, 18, 2)
    c1(31, 18, -1, 0, -1)
    c1(31, 18, -1, 0, 1)

    ui_specs = [
        (6, True, 16, -128, 1),
        (7, False, 17, -128, 1),
        (8, True, 16, 128, -127),
        (9, False, 17, 128, -127),
    ]
    for out, horizontal, coord, coefficient, bias in ui_specs:
        for d in range(-2, 3):
            dy, dx = (0, d) if horizontal else (d, 0)
            c1(out, 14, 1, dy, dx)
            c1(out, 0, 1, dy, dx)
            c1(out, 18, -1, dy, dx)
        c1(out, coord, coefficient)
        p["b1"][out] = bias

    # Paired probes represent clipped cup-pixel counts.
    for j, (dy, dx) in enumerate(
        ((1, 0), (-1, 0), (0, 1), (0, -1))
    ):
        out = 10 + 2 * j
        for distance in (1, 2):
            for color in (11, 13):
                for channel in (out, out + 1):
                    c1(channel, color, 1,
                       dy * distance, dx * distance)
        p["b1"][out + 1] = -1

    def masked_coord(out, color, coord, complement=False):
        c1(out, color, 2)
        c1(out, coord, -1 if complement else 1)
        p["b1"][out] = -1 if complement else -2

    for base, color in ((18, 9), (22, 6)):
        masked_coord(base, color, 16)
        masked_coord(base + 1, color, 16, True)
        masked_coord(base + 2, color, 17)
        masked_coord(base + 3, color, 17, True)

    for i in range(10):
        c2(i, i, 1)

    def probe(out, pair, dy=0, dx=0):
        c2(out, pair, 1, dy, dx)
        c2(out, pair + 1, -1, dy, dx)

    for out, backing, bdy, bdx, arms in (
        (10, 10, 1, 0, ((16, 0, -1), (14, 0, 1))),
        (11, 12, -1, 0, ((16, 0, -1), (14, 0, 1))),
        (12, 14, 0, 1, ((12, -1, 0), (10, 1, 0))),
        (13, 16, 0, -1, ((12, -1, 0), (10, 1, 0))),
    ):
        probe(out, backing, bdy, bdx)
        for pair, dy, dx in arms:
            probe(out, pair, dy, dx)
        c2(out, 27, 1)
        p["b2"][out] = -3

    for out, inp in enumerate(range(18, 26), 14):
        c2(out, inp, 1)

    for out, dy, dx in (
        (22, 0, -1), (23, 0, 1),
        (24, -1, 0), (25, 1, 0),
    ):
        c2(out, 1, 1)
        c2(out, 1, -1, dy, dx)

    adjacent_pairs = [
        ((0, -1), (1, 0)),
        ((-1, 0), (0, -1)),
        ((-1, 0), (0, 1)),
        ((0, 1), (1, 0)),
    ]
    for out, neighbors in enumerate(adjacent_pairs, 26):
        c2(out, 27, 1)
        for dy, dx in neighbors:
            c2(out, 1, 1, dy, dx)
        p["b2"][out] = -2

    c2(30, 0, 1)
    c2(30, 1, -1)
    c2(30, 26, 1)
    c2(31, 28, 1)
    c2(31, 31, 1)

    top = {col(6): 1}
    bottom = {col(8): 1}
    left_ui = {row(7): 1}
    right_ui = {row(9): 1}
    horizontal_ui = merge(top, bottom)
    vertical_ui = merge(left_ui, right_ui)

    # Platform geometry and gravity-normalized depth.
    feature(0, merge({0: 1}, horizontal_ui), -1)
    feature(1, {0: 2, 4: -1, col(8): 2}, -3)
    feature(2, merge({22: 1, 23: 1}, horizontal_ui), -1)
    feature(4, {0: 2, 4: 1, col(6): 2}, -4)
    feature(5, {10: 1, 11: 1})
    feature(6, horizontal_ui)
    feature(7, {30: 1, 0: -1, 1: 1})

    feature(3, merge(
        {12: 2, 13: 2, 4: -1},
        {i: 2 * v for i, v in vertical_ui.items()},
    ), -3)

    for out, inp in enumerate(range(14, 22), 8):
        feature(out, {inp: 1})

    side = {12: 2, 13: 2}
    twice_h = {i: 2 * v for i, v in horizontal_ui.items()}
    feature(16, merge(side, twice_h, {4: 1}), -4)
    feature(17, merge(side, twice_h, {4: -1}), -3)
    feature(18, merge(side, twice_h, {5: 1}), -4)
    feature(19, merge(side, twice_h, {5: -1}), -3)

    feature(20, top)
    feature(21, left_ui)
    feature(22, bottom)
    feature(23, right_ui)
    feature(24, {i: 1 for i in range(26, 30)})
    feature(25, {30: 1, 1: 1})

    feature(26, merge({
        22: 1, col(3): 1, col(30): -1,
    }, horizontal_ui), -2)
    feature(27, merge({
        23: 1, col(3): 1, col(30): -1,
    }, horizontal_ui), -2)

    # Visible walls and lateral display boundaries.
    feature(28, merge(
        {31: 1},
        {i: 2 * v for i, v in horizontal_ui.items()},
    ), -2)

    feature(30, {
        20: 1, 3: 2, col(0): -3, col(30): -3,
    }, -2)
    feature(31, {
        21: 1, 3: 2, col(0): -3, col(30): -3,
    }, -2)

    gate = 32.0
    eps = 1 / 64.0
    gain = 256.0
    threshold = 1.964

    # Two directional distance rays, using clipped channel pairs.
    for pair, dx in ((0, -1), (2, 1)):
        for out in (pair, pair + 1):
            ch(out, pair, 1, 0, dx)
            ch(out, pair + 1, -1, 0, dx)
            for safe_pair in (4, 6):
                ch(out, safe_pair, 1 + eps)
                ch(out, safe_pair + 1, -(1 + eps))
            p["Uh"][out, 0] = -gate
            p["Uh"][out, 6] = gate
            p["Uh"][out, 28] = 1 + eps
            p["bh"][out] = -gate - eps
        p["bh"][pair + 1] -= 1

    # Transport endpoint quality along platform interiors.
    for pair, dx in ((8, -1), (10, 1)):
        for out in (pair, pair + 1):
            ch(out, pair, 1, 0, dx)
            ch(out, pair + 1, -1, 0, dx)
            for ray in (0, 2):
                ch(out, ray, 1, 0, dx)
                ch(out, ray + 1, -1, 0, dx)
            p["Uh"][out, 0] = gate
            p["Uh"][out, 28] = 1
            p["bh"][out] = -gate
        p["bh"][pair + 1] -= 1

    # Propagate safe landing regions against gravity.
    for pair, dy, orientation in ((4, 1, 20), (6, -1, 22)):
        for out in (pair, pair + 1):
            ch(out, pair, 1, dy, 0)
            ch(out, pair + 1, -1, dy, 0)
            for quality in (8, 10):
                ch(out, quality, gain)
                ch(out, quality + 1, -gain)
            p["Uh"][out, 0] = -gain * threshold
            p["Uh"][out, 5] = 1
            p["Uh"][out, orientation] = gate
            p["bh"][out] = -gate
        p["bh"][pair + 1] -= 1

    # Per-platform certificate.
    for out in (12, 13):
        for quality in (8, 10):
            ch(out, quality, gain)
            ch(out, quality + 1, -gain)
        p["Uh"][out, 0] = gate
        p["bh"][out] = -gate - gain * threshold
    p["bh"][13] -= 1

    # Balanced force channels. The baseline cancels in their
    # difference and disappears when the selected piece is safe.
    for out, sign in ((14, 1), (15, -1)):
        for dx in (-1, 1):
            ch(out, 2, sign, 0, dx)
            ch(out, 3, -sign, 0, dx)
            ch(out, 0, -sign, 0, dx)
            ch(out, 1, sign, 0, dx)
        ch(out, 12, -16)
        ch(out, 13, 16)
        p["Uh"][out, 2] = gate
        p["Uh"][out, 28] = -gate
        p["bh"][out] = -gate + 4

    def fm(i):
        return 32 + i

    def fa(i):
        return 64 + i

    def hm(i):
        return 16 + i

    def expr(terms=None, bias=0):
        return dict(terms or {}), float(bias)

    def combine(*items):
        terms, bias = {}, 0.0
        for coefficients, intercept in items:
            bias += intercept
            for i, v in coefficients.items():
                terms[i] = terms.get(i, 0.0) + v
        return terms, bias

    def scale(item, value):
        terms, bias = item
        return ({i: value * v for i, v in terms.items()},
                value * bias)

    index = 0

    def unit(item):
        nonlocal index
        out = index
        index += 1
        terms, bias = item
        for i, v in terms.items():
            p["Wz"][out, i] = v
        p["bz"][out] = bias
        return out

    p["bo"][:] = -1000
    p["bo"][1:5] = [-0.3, -0.3, 0, 0]
    p["bo"][5] = 0.45
    p["bo"][6] = 0.15

    force = expr({hm(14): 512, hm(15): -512})
    uncertainty = expr({hm(14): 64, hm(15): 64})
    corner = unit(expr({fm(24): 1}))

    positive_force = unit(combine(
        force, scale(uncertainty, 0.25),
        expr({fm(2): 20, fm(24): -100, fm(26): -100}, -20),
    ))
    negative_force = unit(combine(
        scale(force, -1), scale(uncertainty, -0.25),
        expr({fm(2): 20, fm(24): -100, fm(27): -100}, -20),
    ))
    p["Wo"][4, positive_force] = 10
    p["Wo"][3, negative_force] = 10

    # Exclusive source interception is a hard movement guard.
    for action, protection in ((4, 26), (3, 27)):
        guard = unit(expr({fm(protection): 1}))
        p["Wo"][action, guard] = -100

    selected_cmid = expr({
        fm(10): 0.5, fm(11): -0.5,
    }, 0.5)
    source_error = combine(
        expr({fm(30): 1}), scale(selected_cmid, -1),
    )
    for action, sign, protection in ((4, 1, 26), (3, -1, 27)):
        activation = expr({
            fm(30): 20, fm(31): 20,
            fm(2): 20, fm(24): -100,
            fm(protection): -100,
        }, -39)
        baseline = unit(activation)
        directed = unit(combine(
            activation, scale(source_error, sign * 8),
        ))
        p["Wo"][action, directed] += 4
        p["Wo"][action, baseline] -= 4

    unsafe = unit(expr({
        fa(0): 1, hm(12): -1, hm(13): 1,
    }))
    p["Wo"][5, unsafe] = -128
    missing = unit(expr({
        fm(30): 1, fm(31): 1,
        fm(20): 10, fm(22): 10,
    }, -10))
    p["Wo"][5, missing] = -1

    corner_present = unit(expr({
        fm(7): 1, fm(20): 1, fm(22): 1,
    }, -1))
    p["Wo"][5, corner_present] = -1.4
    p["Wo"][5, corner] = -2
    p["Wm"][20, 20] = 2
    p["Wmz"][20, corner] = 0.5
    p["Wo"][5, 148] = 1.4

    # A clipped selected-uncertainty indicator.
    status = expr({14: 0.25, 15: 0.25})
    active = unit(status)
    active_excess = unit(combine(status, expr(bias=-1)))

    # Persistent lateral direction, reset on selection and safety.
    for memory, action, opposite in ((22, 4, 3), (23, 3, 4)):
        p["Wm"][memory, memory] = 2
        p["Wma"][memory, action] = 2
        p["Wma"][memory, opposite] = -2
        p["Wma"][memory, 6] = 2
        p["Wmz"][memory, active] = 10
        p["Wmz"][memory, active_excess] = -10
        p["bm"][memory] = -10
        p["Wo"][action, 128 + memory] = 4

    # Away-from-receiver fallback, only for uncertified lines.
    for action, orientation in ((1, 20), (2, 22)):
        ready = unit(combine(
            uncertainty,
            expr({fm(orientation): 32, fm(24): -100}, -32),
        ))
        p["Wo"][action, ready] += 0.6

    upper_guard = unit(expr({
        fm(9): 100, fm(20): 100,
    }, -181))
    lower_guard = unit(expr({
        fm(8): 100, fm(22): 100,
    }, -181))
    p["Wo"][1, upper_guard] = -10
    p["Wo"][2, lower_guard] = -10

    # Conservative alignment fallback for quarter-turn views.
    vertical_error = expr({
        fm(3): -1, fm(8): -0.5, fm(9): 0.5,
    }, 0.5)
    for action, sign in ((2, 1), (1, -1)):
        movement = unit(combine(
            scale(vertical_error, sign * 64),
            expr({
                fm(21): 32, fm(23): 32, fm(24): -100,
            }, -33.7),
        ))
        p["Wo"][action, movement] += 4

    # Corner alignment against an extreme source and side cup.
    corner_errors = [
        (
            expr({fm(14): 1, fm(10): -1}),
            expr({fm(9): 1, fm(17): -1}),
            20,
        ),
        (
            expr({fm(11): 1, fm(15): -1}),
            expr({fm(16): 1, fm(8): -1}),
            22,
        ),
    ]
    for x_error, y_error, orientation in corner_errors:
        activation = expr({
            fm(24): 32, fm(orientation): 32,
        }, -65.7)
        for error, positive, negative in (
            (x_error, 4, 3), (y_error, 2, 1),
        ):
            up = unit(combine(activation, scale(error, 64)))
            down = unit(combine(activation, scale(error, -64)))
            p["Wo"][positive, up] += 4
            p["Wo"][negative, down] += 4

    # Initial selection prefers a lower piece when others exist.
    other_mass = expr({
        fa(0): 4096,
        fa(8): -2048, fa(9): -2048,
        fa(10): -2048, fa(11): -2048,
    })
    other = unit(other_mass)
    other_excess = unit(combine(other_mass, expr(bias=-1)))
    p["Wo"][6, other] += 8
    p["Wo"][6, other_excess] -= 8
    p["Wm"][24, 24] = 100
    p["bm"][24] = 0.01
    p["Wo"][6, 152] -= 8

    # Detect a repeated, unchanged selected-piece position.
    # Small stored coordinates keep tanh approximately linear.
    position = expr({
        fm(8): 0.005, fm(9): -0.005,
        fm(10): 0.0070710678, fm(11): -0.0070710678,
    }, 0.0120710678)
    position_unit = unit(position)
    p["Wmz"][30, position_unit] = 1
    for memory, sign in ((31, 1), (32, -1)):
        p["Wmz"][memory, position_unit] = sign * 10000
        p["Wm"][memory, 30] = -sign * 10000
        p["bm"][memory] = 1
        for action in range(1, 5):
            p["Wo"][action, 128 + memory] = -50

    for i in range(8):
        p["Wma"][i, i] = 2
    for action, opposite in ((1, 2), (2, 1), (3, 4), (4, 3)):
        p["Wo"][action, 128 + opposite] -= 1.5
    p["Wo"][5, 133] -= 2

    # Click score: lower uncertified, unselected pieces first.
    p["Wc"][0, 25] = 10
    p["Wc"][0, 7] = 2
    p["Wc"][0, 1] = 20
    p["Wc"][0, 4] = 20
    for bound in (8, 9, 10, 11):
        p["Wc"][0, bound] -= 50
    for quality in (8, 10):
        p["Wc"][0, 32 + quality] = -4
        p["Wc"][0, 32 + quality + 1] = 4
    p["Wc"][0, 44] = -100
    p["Wc"][0, 45] = 100

    return p
