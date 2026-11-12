import math
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
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    opposite = [1, 0, 3, 2]

    def c1(o, i, dy=0, dx=0, w=1.0):
        p["W1"][o, i, dy + 2, dx + 2] += w

    def c2(o, i, dy=0, dx=0, w=1.0):
        p["W2"][o, i, dy + 1, dx + 1] += w

    def f(o, i, w=1.0):
        p["W3"][o, i] += w

    def z(o, i, w=1.0):
        p["Wz"][o, i] += w

    def fm(c):
        return 32 + c

    def cap(o, a, b, w=1.0):
        f(o, a, w)
        f(o, b, -w)

    def om(a, m, w):
        p["Wo"][a, 128 + m] += w

    # ------------------------------------------------------------
    # Local predicates.
    # ------------------------------------------------------------
    c1(0, 14)
    c1(2, 2)

    # Goal boundary beside visible interior.
    c1(1, 9, w=4.0)
    for dy, dx in dirs:
        c1(1, 2, dy, dx)
    p["b1"][1] = -4.0

    # Configuration-space obstacle masks for a four-pixel object.
    # Channel 18 also supplies the boundary test through zero padding.
    for out, colours in (
        (3, (5, 12)),
        (4, (3, 4, 5, 11, 12, 15)),
    ):
        p["b1"][out] = 16.0
        for dy in range(-1, 3):
            for dx in range(-1, 3):
                c1(out, 18, dy, dx, -1.0)
                for colour in colours:
                    c1(out, colour, dy, dx)

    for colour in (3, 4, 5, 11, 12, 15):
        c1(23, colour)

    # Unique upper-left central anchor of a crate.
    for dy in (0, 1):
        for dx in (0, 1):
            c1(5, 9, dy, dx)
    p["b1"][5] = -3.0

    # Exact player-held and bot-held crate templates.
    for out, border in ((6, 0), (7, 5), (20, 3)):
        for yy in range(4):
            for xx in range(4):
                colour = 9 if yy in (1, 2) and xx in (1, 2) else border
                c1(out, colour, yy - 1, xx - 1)
        p["b1"][out] = -15.0

    rail = (
        (2, 1, 2, 2),
        (1, 2, 2, 2),
        (2, 2, 2, 1),
        (2, 2, 1, 2),
    )
    anchors = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for out, (ay, ax) in zip(range(8, 12), anchors):
        for yy in range(4):
            for xx in range(4):
                c1(out, rail[yy][xx], yy - ay, xx - ax)
        p["b1"][out] = -15.0

    for d, (dy, dx) in enumerate(dirs):
        # Player mask shifted two pixels.
        c1(12 + d, 14, -2 * dy, -2 * dx)

        # A carried-crate centre pair followed by its black edge.
        # The subsequent layers complete a four-pixel lookahead.
        py, px = (0, 1) if dy else (1, 0)
        for distance in (1, 2):
            c1(16 + d, 9, -distance * dy, -distance * dx)
            c1(16 + d, 9,
               -distance * dy + py, -distance * dx + px)
        c1(16 + d, 0)
        p["b1"][16 + d] = -4.0

    c1(21, 11)

    # Pink-bot centre: also valid when its border is highlighted.
    for dy in (0, 1):
        for dx in (0, 1):
            c1(22, 15, dy, dx)
    p["b1"][22] = -3.0

    # Vertical player-facing predicates.
    c1(24, 0)
    c1(24, 14, 1, 0)
    p["b1"][24] = -1.0
    c1(25, 0)
    c1(25, 14, -1, 0)
    p["b1"][25] = -1.0

    # A crate adjoining a goal's internal 9/2 seam.
    # Such seams distinguish deposited crates from crates outside it.
    seam_specs = [
        (26, (1, 2), ((0, 4, 9), (1, 4, 2))),
        (27, (1, 1), ((0, -1, 9), (1, -1, 2))),
        (28, (2, 1), ((4, 0, 9), (4, 1, 2))),
        (29, (1, 1), ((-1, 0, 9), (-1, 1, 2))),
    ]
    for out, (ay, ax), seam in seam_specs:
        for yy in (1, 2):
            for xx in (1, 2):
                c1(out, 9, yy - ay, xx - ax)
        for yy, xx, colour in seam:
            c1(out, colour, yy - ay, xx - ax)
        p["b1"][out] = -5.0

    # Three consecutive 9s cannot be a crate's 2x2 centre.
    for offset in (-1, 0, 1):
        c1(30, 9, 0, offset)
        c1(31, 9, offset, 0)
    p["b1"][30] = -2.0
    p["b1"][31] = -2.0

    # ------------------------------------------------------------
    # Second convolution: bounded masks and aligned anchors.
    # ------------------------------------------------------------
    c2(0, 0)
    c2(2, 2)

    goal_terms = [
        (1, 0, 0), (30, 0, 0), (31, 0, 0),
        (26, 0, 1), (27, 0, 0),
        (28, 1, 0), (29, 0, 0),
    ]
    for out in (1, 26):
        for inp, dy, dx in goal_terms:
            c2(out, inp, dy, dx)
    p["b2"][26] = -1.0

    for a, b, inp in ((3, 4, 3), (5, 6, 4)):
        c2(a, inp)
        c2(b, inp)
        p["b2"][b] = -1.0

    def expanded_pair(a, b, inputs):
        for inp in inputs:
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    c2(a, inp, dy, dx)
                    c2(b, inp, dy, dx)
        p["b2"][b] = -1.0

    expanded_pair(7, 8, [5])
    expanded_pair(9, 10, [6])
    expanded_pair(11, 12, [7])
    expanded_pair(13, 14, range(8, 12))

    for d, (dy, dx) in enumerate(dirs):
        c2(15 + d, 12 + d, -dy, -dx)
        c2(19 + d, 16 + d, -dy, -dx)

    c2(23, 20)
    c2(24, 21)
    c2(25, 22)
    c2(27, 5)
    c2(28, 24)
    c2(28, 25)
    c2(30, 23)

    # ------------------------------------------------------------
    # Axial features. Max(count)-max(count-1) bounds each extent.
    # ------------------------------------------------------------
    cap(0, 3, 4)
    cap(1, 5, 6)
    cap(2, 7, 8)
    cap(3, 9, 10)
    cap(7, 13, 14)

    def goal_row(out, weight=1.0):
        f(out, 32 + 1, weight)
        f(out, 32 + 26, -weight)

    def goal_col(out, weight=1.0):
        f(out, 64 + 1, weight)
        f(out, 64 + 26, -weight)

    goal_row(4)
    goal_col(5)

    f(6, 2)
    goal_row(6)
    goal_col(6)
    p["b3"][6] = -2.0

    # Pickup seeds use one anchor, not a partly occluded footprint.
    f(8, 27)
    f(8, 25, 2.0)
    cap(8, 9, 10, -1.0)
    cap(8, 3, 4, -1.0)
    goal_row(8, -0.5)
    goal_col(8, -0.5)

    f(10, 27)
    goal_row(10)
    goal_col(10)
    p["b3"][10] = -2.0

    cap(11, 9, 10)
    goal_row(11)
    goal_col(11)
    p["b3"][11] = -2.0

    # Bounded available-crate interaction flag.
    for out in (12, 9):
        f(out, 23, 2.0)
        goal_row(out, -1.0)
        goal_col(out, -1.0)
    p["b3"][9] = -1.0

    f(13, 24)

    # Rail release is local to the carried crate.
    cap(14, 9, 10)
    cap(14, 13, 14)
    p["b3"][14] = -1.0

    for d in range(4):
        f(15 + d, 15 + d)
        f(19 + d, 19 + d)

        # Goal-only crossing prediction: no rail/global-region OR.
        f(23 + d, 19 + d)
        goal_row(23 + d, 0.5)
        goal_col(23 + d, 0.5)
        p["b3"][23 + d] = -1.0

        f(27 + d, 15 + d)
        f(27 + d, 30)
        cap(27 + d, 13, 14)
        p["b3"][27 + d] = -1.0

    f(31, 28)

    # ------------------------------------------------------------
    # Two bounded flood fills and their arrival-time accumulators.
    # ------------------------------------------------------------
    def fill(a, b, counter):
        for dy, dx in dirs:
            for out in (a, b):
                p["Wh"][out, a, dy + 1, dx + 1] = 1.0
                p["Wh"][out, b, dy + 1, dx + 1] = -1.0
        p["Wh"][counter, counter, 1, 1] = 1.0
        p["Wh"][counter, a, 1, 1] = 1.0
        p["Wh"][counter, b, 1, 1] = -1.0

    fill(0, 1, 2)
    fill(3, 4, 5)

    p["Uh"][0, 8] = 1.0
    p["Uh"][0, 3] = 1001.0
    p["Uh"][0, 0] = -1000.0
    p["Uh"][0, 7] = -1000.0
    p["Uh"][0, 10] = -1000.0
    p["Uh"][1] = p["Uh"][0]
    p["bh"][1] = -1.0

    p["Uh"][3, 6] = 1.0
    p["Uh"][3, 7] = 1.0e-8
    p["Uh"][3, 1] = -1000.0
    p["Uh"][4] = p["Uh"][3]
    p["bh"][4] = -1.0

    for d, (dy, dx) in enumerate(dirs):
        p["Wh"][6 + d, 2, dy + 1, dx + 1] = 1.0
        p["Uh"][6 + d, 15 + d] = 1000.0
        p["bh"][6 + d] = -1000.0

        p["Wh"][10 + d, 5, dy + 1, dx + 1] = 1.0
        p["Uh"][10 + d, 19 + d] = 1000.0
        p["bh"][10 + d] = -1000.0

    p["Wh"][14, 5, 1, 1] = 1.0
    p["Uh"][14, 3] = 1000.0
    p["bh"][14] = -1000.0

    # ------------------------------------------------------------
    # Decision features.
    # ------------------------------------------------------------
    for d in range(4):
        z(d, 6 + d)
        z(d, fm(3), -256.0)

        z(4 + d, 10 + d)
        z(4 + d, fm(3), 256.0)
        p["bz"][4 + d] = -256.0

        z(8 + d, fm(27 + d))
        z(8 + d, fm(3), 2.0)
        p["bz"][8 + d] = -2.0

        z(12 + d, 6 + d, -1.0)
        z(12 + d, fm(3), 128.0)
        p["bz"][12 + d] = -127.0

        z(20 + d, fm(23 + d))

        z(24 + d, 10 + d)
        z(24 + d, 14, -1.0)
        z(24 + d, fm(27 + d), -400.0)

        # Empty-handed collision cost, with an exception for turning
        # into an immediately adjacent pickup target.
        z(36 + d, fm(27 + d))
        z(36 + d, fm(3), -2.0)

        z(40 + d, fm(27 + d))
        z(40 + d, 6 + d, 0.5)
        z(40 + d, fm(3), -256.0)
        p["bz"][40 + d] = -63.0

    z(16, fm(3))
    z(17, fm(11))
    z(18, fm(12))
    z(18, fm(9), -1.0)
    z(18, fm(3), -2.0)
    z(19, fm(13))
    z(19, fm(3), -2.0)
    z(30, fm(14))
    z(32, 14)
    z(33, fm(31))

    # ------------------------------------------------------------
    # Memory: crossing confirmation, action history, regrip intent.
    # ------------------------------------------------------------
    for d in range(4):
        p["Wmz"][d, 20 + d] = 4.0
        p["bm"][d] = -2.0

        p["Wm"][4 + d, d] = 8.0
        p["Wma"][4 + d, 1 + d] = 8.0
        p["bm"][4 + d] = -12.0

        p["Wma"][8 + d, 1 + d] = 1.0

    p["Wma"][12, 5] = 1.0

    # Remember whether a release would be a regrip rather than delivery.
    p["Wmz"][13, 16] = 5.0
    p["Wmz"][13, 17] = -8.0
    p["Wmz"][13, 30] = -8.0
    p["bm"][13] = -2.5

    p["Wmz"][14, 33] = 5.0
    p["bm"][14] = -2.5

    for out, orientation_weight in ((15, 8.0), (16, -8.0)):
        p["Wm"][out, 13] = 8.0
        p["Wm"][out, 14] = orientation_weight
        p["Wma"][out, 5] = 8.0
        p["bm"][out] = -20.0

    # ------------------------------------------------------------
    # Action logits.
    # ------------------------------------------------------------
    p["bo"][:] = -1.0e6

    for d in range(4):
        a = 1 + d
        p["bo"][a] = 37.5
        p["Wo"][a, d] = 1.0
        p["Wo"][a, 4 + d] = 1.0
        p["Wo"][a, 8 + d] = -400.0
        p["Wo"][a, 12 + d] = -400.0
        p["Wo"][a, 32] = -1.0
        p["Wo"][a, 36 + d] = -400.0
        p["Wo"][a, 40 + d] = 400.0

        om(a, 8 + d, 0.03)
        om(a, 8 + opposite[d], -0.02)

        # First regripping move is perpendicular to the old grip.
        om(a, 16 if d < 2 else 15, 37.5)

    p["bo"][5] = 1550.0
    p["Wo"][5, 16] = 850.5
    p["Wo"][5, 17] = 1000.0
    p["Wo"][5, 18] = 1400.0
    p["Wo"][5, 19] = 1400.0
    p["Wo"][5, 30] = 1000.0

    for d in range(4):
        p["Wo"][5, 24 + d] = -120.0
        om(5, 4 + d, 600.0)

    om(5, 12, -800.0 / math.tanh(1.0))

    return p
