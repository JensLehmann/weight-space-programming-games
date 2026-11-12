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
    p = {name: np.zeros(shape, np.float32)
         for name, shape in shapes.items()}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def c1(o, i, v=1.0, dy=0, dx=0):
        p["W1"][o, i, dy + 2, dx + 2] += v

    def c2(o, i, v=1.0, dy=0, dx=0):
        p["W2"][o, i, dy + 1, dx + 1] += v

    def f3(o, terms, bias=0.0):
        for i, v in terms.items():
            p["W3"][o, i] += v
        p["b3"][o] = bias

    def add(a, b, scale=1.0):
        result = dict(a)
        for k, v in b.items():
            result[k] = result.get(k, 0.0) + scale * v
        return result

    # ------------------------------------------------------------
    # Colour planes and logical-cell neighbour samples.
    # ------------------------------------------------------------
    for o, colour in enumerate((4, 8, 3, 0)):
        c1(o, colour)

    for j, (dy, dx) in enumerate(directions):
        # Grey assembly occupancy, excluding selected coloured bodies.
        for colour in (3, 4):
            c1(4 + j, colour, dy=2 * dy, dx=2 * dx)
        c1(8 + j, 0, dy=2 * dy, dx=2 * dx)

    # Exclude the top-row status bar.
    c1(12, 16, -128.0)
    p["b1"][12] = 1.0

    c1(13, 16)
    c1(14, 17)
    c1(15, 16, 0.5)
    c1(15, 17, 0.5)

    for o, coordinate, sign, bias in (
        (16, 16, -1.0, 0.5),
        (17, 16, 1.0, -0.5),
        (18, 17, -1.0, 0.5),
        (19, 17, 1.0, -0.5),
    ):
        c1(o, coordinate, sign)
        p["b1"][o] = bias

    # ------------------------------------------------------------
    # Grey tips, highlighting, and selected green joints.
    # ------------------------------------------------------------
    for o, bias in ((0, -0.5), (1, -1.0)):
        c2(o, 0, 2.0)
        for j, (dy, dx) in enumerate(directions):
            c2(o, 4 + j, -1.0, dy, dx)
        p["b2"][o] = bias

    # Unselected red in black-highlight mode.
    c2(2, 1)
    for j, (dy, dx) in enumerate(directions):
        c2(2, 8 + j, -1.0, dy, dx)

    c2(3, 1)
    c2(3, 2)
    c2(4, 2)

    c2(5, 3)
    c2(5, 12, -1.0)
    c2(6, 0)
    c2(6, 12, -1.0)
    c2(7, 12)

    c2(8, 14)
    c2(9, 13)
    c2(10, 15)
    for i in (16, 17, 18, 19):
        c2(11, i)
    c2(12, 1)

    # Any connector not adjacent to black highlighting.
    c2(13, 1)
    c2(13, 2)
    for j, (dy, dx) in enumerate(directions):
        c2(13, 8 + j, -1.0, dy, dx)

    # A green joint belonging to a selected coloured piece normally
    # has fewer than two grey neighbours. A joint wholly inside the
    # unselected assembly normally has two.
    for o, bias in ((14, -0.5), (15, -1.0)):
        c2(o, 2, 2.0)
        for j, (dy, dx) in enumerate(directions):
            c2(o, 4 + j, -1.0, dy, dx)
        p["b2"][o] = bias

    # ------------------------------------------------------------
    # Endpoint masks and masked coordinate features.
    # ------------------------------------------------------------
    grey_tip = {0: 2.0, 1: -2.0, 7: -2.0}
    selected_green = {14: 2.0, 15: -2.0}
    q_grey = add({12: 1.0}, selected_green)
    t_grey = add(grey_tip, selected_green)
    q_colour = {3: 1.0, 13: -1.0}
    t_colour = {2: 1.0, 4: 1.0}

    def endpoint_group(start, mask):
        f3(start, mask)
        for j, coordinate in enumerate((9, 8, 10), 1):
            f3(start + j, add(mask, {coordinate: 1.0}), -1.0)

    endpoint_group(0, q_grey)
    endpoint_group(4, t_grey)
    endpoint_group(8, q_colour)
    endpoint_group(12, t_colour)

    f3(16, {6: 1.0})              # Grey body.
    f3(17, {5: 1.0})              # Black highlighting / resize marker.
    f3(18, {12: 1.0})             # All visible red.
    f3(19, {4: 1.0})              # All green.
    f3(20, {12: 1.0, 2: -1.0})   # Selected red in colour mode.
    f3(21, grey_tip)
    f3(22, {2: 1.0})
    f3(23, selected_green)
    f3(24, {11: 1.0})
    f3(25, {5: 1.0})
    f3(26, {32 + 5: 1.0})
    f3(27, {64 + 5: 1.0})

    # ------------------------------------------------------------
    # Tied block.
    #
    # h0-h1: global black-highlighting flag.
    # h2:    colour-mode target seed.
    # h3-h4: clipped expanding wave.
    # h5:    integrated arrival time, hence an attraction field.
    # h6:10: field samples at query endpoints.
    # h11-h12: grey connectivity flood from green joints.
    # h13:   detached grey tips.
    # ------------------------------------------------------------
    W = p["Wh"]
    U = p["Uh"]
    B = p["bh"]

    def hc(o, i, v=1.0, dy=0, dx=0):
        W[o, i, dy + 1, dx + 1] += v

    for o in (0, 1):
        for dx in (-1, 1):
            hc(o, 0, 1.0, dx=dx)
            hc(o, 1, -1.0, dx=dx)
        U[o, 26] = 1.0
        U[o, 27] = 1.0
    B[1] = -1.0

    U[2, 22] = 1.0
    hc(2, 0)
    hc(2, 1, -1.0)
    B[2] = -1.0

    # The difference of these two channels is a Boolean wavefront.
    for o in (3, 4):
        for dy, dx in directions:
            hc(o, 3, 1.0, dy, dx)
            hc(o, 4, -1.0, dy, dx)
        hc(o, 2)
        U[o, 21] = 1.0
        U[o, 23] = 1.0
    B[4] = -1.0

    # Each reached pixel accumulates one per iteration.
    hc(5, 5)
    hc(5, 3)
    hc(5, 4, -1.0)

    gate = 256.0
    for j, (dy, dx) in enumerate(directions):
        hc(6 + j, 5, 1.0, dy, dx)
        U[6 + j, 0] = gate
        B[6 + j] = -gate

    hc(10, 5)
    U[10, 0] = gate
    B[10] = -gate

    # Flood only through grey pixels and green joints.
    for o in (11, 12):
        for dy, dx in directions:
            hc(o, 11, 1.0, dy, dx)
            hc(o, 12, -1.0, dy, dx)
        U[o, 16] = 4.0
        U[o, 19] = 5.0
    B[11] = -4.0
    B[12] = -5.0

    U[13, 21] = 1.0
    hc(13, 11, -1.0)
    hc(13, 12, 1.0)

    # ------------------------------------------------------------
    # Pooled features and geometric policy.
    # ------------------------------------------------------------
    def hm(c):
        return 16 + c

    def fm(c):
        return 32 + c

    def fa(c):
        return 64 + c

    nz = 0

    def zunit(terms=None, bias=0.0):
        nonlocal nz
        o = nz
        nz += 1
        if terms:
            for i, v in terms.items():
                p["Wz"][o, i] += v
        p["bz"][o] = bias
        return o

    def capped(action, terms, bias, ceiling, gain):
        lo = zunit(terms, bias)
        hi = zunit(terms, bias - ceiling)
        p["Wo"][action, lo] += gain
        p["Wo"][action, hi] -= gain
        return lo, hi

    p["bo"][:] = -1000.0
    p["bo"][1:5] = 0.0
    p["bo"][5] = 0.4
    p["bo"][6] = -6.0

    count_scale = 4096.0 / 9.0
    pair_mean_scale = 4096.0 / 18.0

    # Directional field gains. Colour mode also sees stationary red
    # endpoints; compensate their one-pixel self-attraction loss.
    for grey_mode in (True, False):
        for j, action in enumerate((1, 2, 3, 4)):
            for sign in (1.0, -1.0):
                terms = {
                    hm(6 + j): sign * count_scale,
                    hm(10): -sign * count_scale,
                    fm(16): 1000.0 if grey_mode else -1000.0,
                }
                if not grey_mode:
                    terms[fa(22)] = sign * count_scale / 3.0
                capped(
                    action, terms,
                    -1000.0 if grey_mode else 0.0,
                    3.0, 6.0 * sign,
                )

    # Absolute deviation from exactly two endpoints.
    count_error = {}
    for group in (0, 4, 8, 12):
        count_error[group] = (
            zunit({fa(group): count_scale}, -2.0),
            zunit({fa(group): -count_scale}, 2.0),
        )
    grey_flag = zunit({fm(16): 1.0})

    next_logic_cell = 16

    def gated_logic(features, q, t, grey_mode, gain, action):
        nonlocal next_logic_cell
        cell = next_logic_cell
        next_logic_cell += 1

        p["bm"][cell] = -4.0
        for feature, coefficient in features:
            p["Wmz"][cell, feature] += coefficient
        for group in (q, t):
            for feature in count_error[group]:
                p["Wmz"][cell, feature] -= 12000.0

        if grey_mode:
            p["Wmz"][cell, grey_flag] += 10000.0
            p["bm"][cell] -= 10000.0
        else:
            p["Wmz"][cell, grey_flag] -= 10000.0

        p["Wo"][action, 128 + cell] += gain
        p["bo"][action] += gain

    # Exact pair-centre and pair-orientation alignment.
    for q, t, grey_mode in ((0, 4, True), (8, 12, False)):
        for action, coordinate, sign in (
            (1, 1, -1.0), (2, 1, 1.0),
            (3, 2, -1.0), (4, 2, 1.0),
        ):
            delta = zunit({
                fa(t + coordinate): sign * pair_mean_scale,
                fa(q + coordinate): -sign * pair_mean_scale,
            }, -0.018)
            gated_logic(
                [(delta, 2000.0)], q, t, grey_mode, 8.0, action
            )

        errors = []
        for coordinate in (1, 2, 3):
            error = {
                fm(q + coordinate): 1.0,
                fa(q + coordinate): -pair_mean_scale,
                fm(t + coordinate): -1.0,
                fa(t + coordinate): pair_mean_scale,
            }
            for sign in (1.0, -1.0):
                feature = zunit(
                    {k: sign * v for k, v in error.items()}, -0.004
                )
                errors.append((feature, 1500.0))
        gated_logic(errors, q, t, grey_mode, 22.0, 5)

    expandable = zunit({fm(17): 1.0, fm(16): 1.0}, -1.5)
    p["Wo"][5, expandable] = 2000.0

    # Keep large hubs stationary after expanding them.
    hub = zunit({fa(0): count_scale, fm(16): 20.0}, -23.5)
    p["Wo"][6, hub] = 100.0

    done_grey = zunit({
        fm(21): 1.0, fm(18): -1.0, fm(16): 2.0,
    }, -2.0)
    done_colour = zunit({
        fm(22): 1.0, fm(20): -1.0, fm(16): -2.0,
    })
    p["Wo"][6, done_grey] = 100.0
    p["Wo"][6, done_colour] = 100.0

    no_query_grey = zunit({fm(0): -1.0, fm(16): 2.0}, -1.0)
    no_query_colour = zunit({fm(8): -1.0, fm(16): -2.0}, 1.0)
    p["Wo"][6, no_query_grey] = 100.0
    p["Wo"][6, no_query_colour] = 100.0

    # When overlap hides every target, translate to reveal the board
    # rather than repeatedly rotating over the same hidden endpoints.
    no_target_grey = zunit({fm(4): -1.0, fm(16): 1.0})
    no_target_colour = zunit({fm(12): -1.0, fm(16): -1.0}, 1.0)
    p["Wo"][2, no_target_grey] = 60.0
    p["Wo"][2, no_target_colour] = 60.0

    three_lo = zunit({fa(0): count_scale, fm(16): 20.0}, -22.0)
    three_hi = zunit({fa(0): count_scale, fm(16): 20.0}, -23.0)

    green_count = zunit({fa(19): count_scale})
    remaining_lo = zunit({
        fa(18): count_scale, fa(21): count_scale,
    }, -2.0)
    remaining_hi = zunit({
        fa(18): count_scale, fa(21): count_scale,
    }, -3.0)

    # ------------------------------------------------------------
    # Memory: retries, persistence, new-joint detection, initial reset.
    # ------------------------------------------------------------
    p["Wm"][0, 0] = 0.98
    p["Wma"][0, 5] = 0.02
    p["Wma"][0, 6] = -0.08

    p["Wm"][1, 0] = 250.0
    p["Wma"][1, 6] = -50.0
    p["bm"][1] = -21.25
    p["Wo"][6, 129] = 25.0
    p["bo"][6] += 25.0

    # Bounded persistence for three-terminal pieces.
    for j, action in enumerate((1, 2, 3, 4)):
        cell = 2 + j
        p["bm"][cell] = -12.0
        p["Wma"][cell, action] = 8.0
        p["Wmz"][cell, three_lo] = 8.0
        p["Wmz"][cell, three_hi] = -8.0
        p["Wo"][action, 128 + cell] = 2.0
        p["bo"][action] += 2.0

    for action in range(8):
        p["Wma"][6 + action, action] = 3.0
    for action in range(1, 5):
        p["Wo"][action, 128 + 6 + action] = 0.025

    # Escape prolonged movement runs without disturbing short paths.
    p["Wm"][14, 14] = 0.98
    p["Wma"][14, 1:5] = 0.02
    p["Wma"][14, 5] = -0.2
    p["Wma"][14, 6] = -0.2
    p["Wm"][15, 14] = 30.0
    p["bm"][15] = -9.0
    p["Wo"][5, 143] = 12.0
    p["bo"][5] += 12.0

    # Store total green count on a small scale. A newly formed joint
    # can freeze the current subassembly and select a detached piece,
    # but not when only the final endpoint pair remains.
    p["Wmz"][54, green_count] = 0.03
    p["Wmz"][55, green_count] = 30.0
    p["Wm"][55, 54] = -1000.0
    p["bm"][55] = -115.0
    p["Wmz"][55, remaining_lo] = 100.0
    p["Wmz"][55, remaining_hi] = -100.0
    p["Wo"][6, 128 + 55] = 25.0
    p["bo"][6] += 25.0

    # One reset at episode start. These cells do not depend on layout.
    p["bm"][60] = 3.0
    p["Wm"][61, 60] = 100.0
    p["bm"][61] = -4.0
    p["bo"][0] = -100.0
    p["Wo"][0, 128 + 61] = -2000.0

    # ------------------------------------------------------------
    # Click map: prefer detached grey terminals over attached ones.
    # ------------------------------------------------------------
    p["Wc"][0, 21] = 100.0
    p["Wc"][0, 32 + 13] = 60.0
    p["Wc"][0, 22] = 80.0
    p["Wc"][0, 16] = 1.0
    p["Wc"][0, 24] = 0.2

    return p
