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
    colours = (8, 9, 12, 14)

    def w1(out, channels, dy=0, dx=0, weight=1.0):
        if isinstance(channels, int):
            channels = (channels,)
        for c in channels:
            p["W1"][out, c, dy + 2, dx + 2] += weight

    # Upper-left interior anchors of coloured blocks.
    for c, colour in enumerate(colours):
        for dy, dx in ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0)):
            w1(c, colour, dy, dx)
        w1(c, colour, -1, 0, -5)
        w1(c, colour, 0, -1, -5)
        p["b1"][c] = -4

    # Selected rails at block anchors.
    w1(4, 1, 2, -1)
    w1(5, 1, -1, 1)

    # Rails of either selection state.
    w1(6, (2, 3), 1, -1)
    w1(6, (1, 2), 2, -1)
    p["b1"][6] = -1
    w1(7, (1, 2), -1, 1)
    w1(7, (2, 3), -1, 2)
    p["b1"][7] = -1

    w1(8, 0, 1, 1)
    w1(9, 16, weight=63)
    p["b1"][9] = -52
    w1(10, 16, weight=-63)
    p["b1"][10] = 54

    # Selected handle anchors.
    for out in (11, 12):
        w1(out, (6, 15))
        w1(out, (6, 15), -1, -1)
        w1(out, 0, 0, -2)
        w1(out, 0, -2, 0)
        p["b1"][out] = -4
    w1(11, 1, -1, 2)
    w1(12, 1, 2, 0)

    # Unselected clickable handle.
    w1(13, (6, 15))
    w1(13, (6, 15), -1, -1)
    w1(13, 4, 0, -2)
    w1(13, 4, -2, 0)
    p["b1"][13] = -3

    w1(14, 1)
    w1(15, 16)
    w1(16, 17)
    p["b1"][17] = 1

    # Upper-left corner of a black obstacle inside the board.
    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        w1(18, 5, dy, dx)
    w1(18, 4, -1, 0)
    w1(18, 4, 0, -1)
    p["b1"][18] = -5

    def f2(out, terms, bias=0):
        for c, v in terms.items():
            p["W2"][out, c, 1, 1] += v
        p["b2"][out] = bias

    for c in range(4):
        f2(c, {c: 1, 4: -1, 5: -1, 9: -2})
        f2(4 + c, {c: 1, 4: 1, 8: -1, 10: -2}, -1)

    f2(8, {11: 1, 9: -2})
    # Second horizontal-handle anchor; coordinate maxima are unchanged.
    p["W2"][8, 11, 2, 1] = 1
    f2(9, {12: 1, 9: -2})
    f2(10, {14: 1, 9: -2})
    f2(11, {15: 1})
    f2(12, {16: 1})

    blocks = {c: 1 for c in range(4)}

    terms = dict(blocks)
    terms.update({4: 1, 7: 1, 9: -2})
    f2(13, terms, -2)

    terms = dict(blocks)
    terms.update({5: 1, 6: 1, 9: -2})
    f2(14, terms, -2)

    f2(15, {13: 1, 9: -2})

    terms = dict(blocks)
    terms.update({4: 1, 10: -2})
    f2(16, terms, -1)

    terms = dict(blocks)
    terms.update({4: 1, 5: 1, 9: -2})
    f2(17, terms, -1)

    terms = dict(blocks)
    terms[9] = -2
    f2(18, terms)

    terms = dict(blocks)
    terms.update({6: 1, 4: -1, 9: -2})
    f2(19, terms, -1)

    # Obstacle anchor uses the same +1,+1 convention as block anchors.
    p["W2"][20, 18, 0, 0] = 1
    p["W2"][22, 18, 0, 0] = 2
    p["W2"][22, 15, 1, 1] = 1
    p["W2"][22, 16, 1, 1] = 1
    p["b2"][22] = -2

    def f3(out, terms, bias=0):
        for c, v in terms.items():
            p["W3"][out, c] += v
        p["b3"][out] = bias

    for c in range(4):
        f3(c, {4 + c: 1, 12: -1})
        f3(4 + c, {c: 1, 11: 1}, -1)
        f3(8 + c, {c: 1, 12: -1})
        f3(12 + c, {c: 1, 12: 1}, -1)
        f3(16 + c, {c: 1, 11: -1})
        # Ordinary collection excludes an obstructed row.
        for out in (4 + c, 8 + c, 12 + c, 16 + c):
            p["W3"][out, 32 + 20] = -2

    f3(20, {8: 1, 9: 1, 12: 1}, -1)
    f3(21, {8: 1, 9: 1, 11: 1}, -1)
    f3(22, {10: 1, 12: 1}, -1)
    f3(23, {10: 1, 11: 1}, -1)
    f3(24, {8: 1})
    f3(25, {9: 1})
    f3(26, {13: 1})
    f3(27, {14: 1})
    f3(28, {15: 1})

    # Other-carrier coordinates, also encoding obstacle coordinates.
    f3(29, {19: 1, 20: 1, 12: 1}, -1)
    f3(30, {19: 1, 12: -1, 22: 1})

    # Spatial marker code: block=1, carried block=2, obstacle=4.
    f3(31, {18: 1, 17: 1, 20: 4})
    p["Wc"][0, 28] = 1

    def wh(out, inp, dy=0, dx=0, value=1):
        p["Wh"][out, inp, dy + 1, dx + 1] += value

    # Decode spatial markers.
    p["Uh"][0, 31] = 1
    p["bh"][0] = -3
    p["Uh"][1, 31] = 1
    p["bh"][1] = -1
    wh(1, 0, value=-4)
    p["Uh"][2, 31] = 1
    wh(2, 0, value=-4)
    wh(2, 1, value=-1)

    # Horizontal rays and upper/lower right quadrants from an obstacle.
    wh(4, 4, dx=-1)
    wh(4, 0)
    wh(5, 5, dy=1)
    wh(5, 4)
    wh(6, 6, dy=-1)
    wh(6, 4)

    # Blocks at or to the left of the obstacle column.
    wh(7, 2)
    wh(7, 5, dx=-1, value=-1)
    wh(7, 6, dx=-1, value=-1)
    wh(7, 4, dx=-1)

    # Upward projections of selected rail pixels.
    p["Uh"][8, 22] = 1
    wh(8, 8, dy=1)
    wh(9, 2)
    wh(9, 8, value=-100)

    # A block exported strictly above and right of the obstacle.
    wh(10, 2)
    wh(10, 5, dy=1, dx=-1)
    p["bh"][10] = -1

    # Left ray and upper-left quadrant.
    wh(12, 12, dx=1)
    wh(12, 0)
    wh(13, 13, dy=1)
    wh(13, 12)

    # Horizontal handle aligned with the obstacle row.
    p["Uh"][3, 24] = 1
    wh(3, 4, dy=-1)
    wh(3, 12, dy=-1)
    wh(3, 0, dy=-1, value=-1)
    p["bh"][3] = -1

    p["Uh"][14, 24] = 1
    wh(14, 13)
    p["bh"][14] = -1

    # Selected rail extending into the obstacle's column region.
    p["Uh"][15, 22] = 1
    wh(15, 5)
    wh(15, 6)
    wh(15, 4, value=-1)
    p["bh"][15] = -1

    def G(c):
        return 32 + c

    nz = 0

    def zunit(terms, bias=0):
        nonlocal nz
        z = nz
        nz += 1
        for c, v in terms.items():
            p["Wz"][z, c] += v
        p["bz"][z] = bias
        return z

    flags = {
        name: zunit({G(c): 1})
        for name, c in (
            ("H", 24), ("V", 25), ("PH", 26),
            ("PV", 27), ("click", 28)
        )
    }
    sx = zunit({G(20): 1, G(22): 1})
    sy = zunit({G(21): 1, G(23): 1})
    priorities_sum = zunit({G(c): 1 for c in range(4)})

    comparisons = {}
    absent_goal = []
    absent_external = []
    geometry = []

    for c in range(4):
        comparisons[c] = [
            zunit({G(d): 1, G(c): -1})
            for d in range(4) if d != c
        ]
        absent_goal.append(zunit({G(c): -1}, 0.02))
        absent_external.append(
            zunit({G(4 + c): -1, G(16 + c): -1}, 0.5)
        )

        hp = zunit({G(4 + c): 1, G(21): -1}, 2 / 63)
        hn = zunit({G(4 + c): -1, G(21): 1}, -2 / 63)
        vp = zunit({G(12 + c): 1, G(20): -1}, 2 / 63)
        vn = zunit({G(12 + c): -1, G(20): 1}, -2 / 63)
        lp = zunit({G(8 + c): -1, G(20): -1}, 1 + 2 / 63)
        ln = zunit({G(8 + c): 1, G(20): 1}, -1 - 2 / 63)

        hu = zunit(
            {G(22): 1, G(8 + c): 1}, -1 - 4 / 63 + 0.04
        )
        vu = zunit(
            {G(23): 1, G(16 + c): 1}, -1 - 4 / 63 + 0.04
        )
        hr = zunit(
            {G(22): -1, G(8 + c): -1}, 1 + 4 / 63 - 0.02
        )
        vr = zunit(
            {G(23): -1, G(16 + c): -1}, 1 + 4 / 63 - 0.02
        )

        select = {
            G(29): 1, G(30): -1,
            G(8 + c): 1, G(12 + c): -1
        }
        right = zunit(select, 0.001)
        left = zunit({k: -v for k, v in select.items()}, -0.001)
        geometry.append(
            (hp, hn, vp, vn, lp, ln, hu, vu, hr, vr, right, left)
        )

    macro = zunit({0: 1, 10: -1, G(28): -1})
    overlap = zunit(
        {G(22): 1, G(29): -1, 0: 10}, 0.02 - 10
    )
    clear = zunit(
        {G(29): 1, G(22): -1, 0: -10}, 0.025 + 10
    )

    p["bo"][:] = -20
    p["bo"][1:5] = 0
    p["bo"][6] = -0.02
    p["bo"][7] = -5
    nm = 0

    def candidate(action, colour, orientation, terms, bias,
                  require_external=True):
        nonlocal nm
        m = nm
        nm += 1
        coefficients = dict(terms)

        def add(z, value):
            coefficients[z] = coefficients.get(z, 0) + value

        add(flags[orientation], 300)
        bias -= 300
        add(macro, -500)
        for z in comparisons[colour]:
            add(z, -10000)
        add(absent_goal[colour], -10000)
        if require_external:
            add(absent_external[colour], -4)

        for z, value in coefficients.items():
            p["Wmz"][m, z] = 90 * value
        p["bm"][m] = 90 * bias
        p["Wo"][action, 128 + m] = 1
        p["bo"][action] += 1

    for c, geom in enumerate(geometry):
        hp, hn, vp, vn, lp, ln, hu, vu, hr, vr, right, left = geom

        candidate(4, c, "H", {hp: -1, hn: -1}, 0.035)
        candidate(
            1, c, "H",
            {hp: -1, hn: 1, hu: -30, overlap: -30}, -0.025
        )
        candidate(
            2, c, "H",
            {hp: 1, hn: -1, hu: -30, overlap: -30}, -0.025
        )
        candidate(
            3, c, "H",
            {hp: 1, hn: 1, hr: -30, flags["PH"]: -3,
             absent_external[c]: 200},
            -0.025, False
        )
        candidate(
            2, c, "H",
            {hp: 1, hn: 1, hr: -30, flags["PH"]: 3}, -3.025
        )

        candidate(
            2, c, "V", {vp: -1, vn: -1, left: -10000}, 0.035
        )
        candidate(
            3, c, "V",
            {vp: -1, vn: 1, vu: -30, left: -10000}, -0.025
        )
        candidate(
            4, c, "V",
            {vp: 1, vn: -1, vu: -30, left: -10000}, -0.025
        )

        candidate(
            2, c, "V", {lp: -1, ln: -1, right: -10000}, 0.035
        )
        candidate(
            3, c, "V",
            {lp: -1, ln: 1, vu: -30, right: -10000}, -0.025
        )
        candidate(
            4, c, "V",
            {lp: 1, ln: -1, vu: -30, right: -10000}, -0.025
        )

        candidate(
            1, c, "V",
            {vp: 1, vn: 1, vr: -30, flags["PV"]: -3,
             absent_external[c]: 200},
            -0.025, False
        )
        candidate(
            3, c, "V",
            {vp: 1, vn: 1, vr: -30, flags["PV"]: 3}, -3.025
        )

    for c, geom in enumerate(geometry):
        hp, hn = geom[:2]
        candidate(
            3, c, "H", {hp: 1, hn: 1, clear: -30}, -0.025
        )
    assert nm == 56

    # Geometric obstacle controller, gated by spatial predicates.
    def macro_unit(action, terms, bias, weight=1,
                   left=None, carried=None):
        terms = dict(terms)

        def add(key, value):
            terms[key] = terms.get(key, 0) + value

        add(0, 500)
        bias -= 500
        add(10, -500)
        add(G(28), -500)
        if left is not None:
            add(7, 500 if left else -500)
            if left:
                bias -= 500
        if carried is not None:
            add(1, 500 if carried else -500)
            if carried:
                bias -= 500
        z = zunit(terms, bias)
        p["Wo"][action, z] += weight

    # Move the helper load to the adjacent clear row.
    d = {G(30): 63, G(21): -63}
    macro_unit(2, d, 8 - 1, 2, left=True)
    macro_unit(
        1, {k: -v for k, v in d.items()}, -8 - 1, 2, left=True
    )

    # Unload there, then push the detached train across.
    for action, carried in ((3, True), (4, False)):
        for offset, weight in ((1, 10), (0, -20), (-1, 10)):
            macro_unit(
                action, d, 8 + offset, weight,
                left=True, carried=carried
            )

    # Move one row farther out. Cover all but the final paired column,
    # then push toward the obstacle to export blocks into a clear row.
    macro_unit(2, d, 14 - 0.5, 4, left=False)
    macro_unit(
        1, {k: -v for k, v in d.items()},
        -14 + 0.5, 4, left=False
    )
    macro_unit(4, {16 + 9: 4096}, -2.5, 2, left=False)

    # Narrow the carrier before crossing the obstacle row from above.
    macro_unit(3, {15: 100, 14: 500}, -500)

    # Complete a partial load against the obstacle before moving aside.
    macro_unit(
        4, {G(29): 63, G(22): -63, 3: 500},
        -502.5, 10, left=True
    )

    # Nearly linear coordinate memories.
    store_x, store_y = 60, 61
    scale = 0.02
    p["Wmz"][store_x, sx] = scale
    p["Wmz"][store_y, sy] = scale

    # Signed progress tests for the previous movement.
    for action in range(1, 5):
        m = 55 + action
        summary = sy if action in (1, 2) else sx
        stored = store_y if action in (1, 2) else store_x
        direction = -1 if action in (1, 3) else 1

        p["Wmz"][m, summary] = -600 * direction
        p["Wm"][m, stored] = 600 * direction / scale
        p["Wma"][m, action] = 4000
        p["bm"][m] = 600 * 0.012 - 4000

        p["Wo"][action, 128 + m] -= 1.2
        p["bo"][action] -= 1.2
        escape = 3 if action in (1, 2) else 1
        p["Wo"][escape, 128 + m] += 0.7
        p["bo"][escape] += 0.7

    # Switch when the selected carrier's reference is complete.
    p["Wmz"][62, priorities_sum] = -100
    p["Wmz"][62, flags["click"]] = 300
    p["bm"][62] = -298
    p["Wo"][6, 128 + 62] = 1
    p["bo"][6] += 1

    # Recovery switch following a blocked movement.
    for m in range(56, 60):
        p["Wm"][63, m] = 20
    p["Wmz"][63, flags["click"]] = 60

    # At cold start the four progress memories are zero, not their
    # normal inactive value -1. Require a previous action before
    # interpreting them as recovery evidence. For a one-hot previous
    # action these two terms cancel, preserving the ongoing controller.
    p["bm"][63] = -80
    p["Wma"][63, :] = 80

    p["Wo"][6, 128 + 63] = 1.1
    p["bo"][6] += 1.1

    p["bo"][1] += 0.003
    p["bo"][3] += 0.002
    p["bo"][4] += 0.001

    assert nz <= 128
    return p
