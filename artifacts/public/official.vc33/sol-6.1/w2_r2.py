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
    p = {k: np.zeros(shape, dtype=np.float32)
         for k, shape in shapes.items()}

    def c1(o, i, dy=0, dx=0, v=1.0):
        p["W1"][o, i, dy + 2, dx + 2] += v

    def c2(o, i, dy=0, dx=0, v=1.0):
        p["W2"][o, i, dy + 1, dx + 1] += v

    def f(o, i, v=1.0, pool="point"):
        offset = {"point": 0, "row": 32, "col": 64}[pool]
        p["W3"][o, offset + i] += v

    def r(o, i, dy=0, dx=0, v=1.0):
        p["Wh"][o, i, dy + 1, dx + 1] += v

    def d(o, i, v=1.0):
        p["Uh"][o, i] += v

    cardinal = ((-1, 0), (1, 0), (0, -1), (0, 1))
    neighbourhood = tuple(
        (dy, dx)
        for dy in range(-1, 2)
        for dx in range(-1, 2)
    )

    # ----------------------------------------------------------
    # First convolution: colours, player normals, outside probes,
    # structural-wall tests, and bridge endpoints.
    # ----------------------------------------------------------
    for o, colour in ((0, 0), (1, 5), (2, 9), (4, 12)):
        c1(o, colour)

    for colour in (1, 12):
        c1(3, colour)

    for colour in (11, 14):
        c1(5, colour)
        c1(6, colour)

    # Target bases lack the nearby yellow player head.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            c1(6, 4, dy, dx, -1.0)

    # D, U, R, L player-normal predicates.
    for o, head_offset in (
        (7, (-2, 0)),
        (8, (2, 0)),
        (9, (0, -2)),
        (10, (0, 2)),
    ):
        for colour in (11, 14):
            c1(o, colour)
        c1(o, 4, *head_offset)
        p["b1"][o] = -1.0

    # Empty surface probes use background only. In particular,
    # target strips and the yellow progress bar are not probes.
    for o, inward in (
        (11, (1, 0)),
        (12, (-1, 0)),
        (13, (0, 1)),
        (14, (0, -1)),
    ):
        c1(o, 3)
        c1(o, 0, *inward)
        p["b1"][o] = -1.0

    # Vertical and horizontal thin-wall predicates.
    for o, along, outside in (
        (15, ((-2, 0), (2, 0)), (0, -2)),
        (16, ((-2, 0), (2, 0)), (0, 2)),
        (17, ((0, -2), (0, 2)), (-2, 0)),
        (18, ((0, -2), (0, 2)), (2, 0)),
    ):
        c1(o, 5)
        for offset in along:
            c1(o, 5, *offset)
        c1(o, 5, *outside, v=-1.0)
        p["b1"][o] = -2.0

    # Bridge endpoints D, U, R, L.
    for o, behind, ahead in (
        (19, (-2, 0), (1, 0)),
        (20, (2, 0), (-1, 0)),
        (21, (0, -2), (0, 1)),
        (22, (0, 2), (0, -1)),
    ):
        for colour in (1, 12):
            c1(o, colour)
            c1(o, colour, *behind)
            c1(o, colour, *ahead, v=-1.0)
        p["b1"][o] = -1.0

    c1(23, 17)
    c1(24, 16)

    # Boundary controls identify long structural walls.
    for dx in range(-2, 3):
        c1(25, 9, 0, dx)
        c1(29, 9, 0, dx)
    for dy in range(-2, 3):
        c1(26, 9, dy, 0)
        c1(30, 9, dy, 0)

    c1(25, 16, v=-64.0)
    c1(29, 16, v=64.0)
    c1(26, 17, v=64.0)
    c1(30, 17, v=-64.0)
    p["b1"][25] = p["b1"][30] = -0.5
    p["b1"][26] = p["b1"][29] = -64.5

    # ----------------------------------------------------------
    # Second convolution.
    # ----------------------------------------------------------
    for o, i in ((0, 0), (1, 1), (2, 2), (3, 4), (4, 6)):
        c2(o, i)

    c2(5, 5)
    c2(5, 6, v=-1.0)

    # Coordinates masked to player bases.
    for o, coordinate in ((6, 23), (7, 24)):
        c2(o, coordinate)
        c2(o, 5)
        c2(o, 6, v=-1.0)
        p["b2"][o] = -1.0

    # Player edges directly touching their supporting platform.
    for o, normal, inward in (
        (8, 7, (1, 0)),
        (9, 8, (-1, 0)),
        (10, 9, (0, 1)),
        (11, 10, (0, -1)),
    ):
        c2(o, normal)
        c2(o, 0, *inward)
        p["b2"][o] = -1.0

    for o, i in zip(range(12, 16), range(11, 15)):
        c2(o, i)

    # Paired ReLUs clip the wall tests to binary predicates.
    for a, b, inputs in (
        (16, 17, (15, 16)),
        (18, 19, (17, 18)),
    ):
        for i in inputs:
            c2(a, i)
            c2(b, i)
        p["b2"][b] = -1.0

    # Structural walls have a target, bridge, or boundary control.
    for a, b, boundary in (
        (20, 21, (25, 29)),
        (22, 23, (26, 30)),
    ):
        for dy, dx in neighbourhood:
            for i in (3, 6):
                c2(a, i, dy, dx)
                c2(b, i, dy, dx)
            for i in boundary:
                c2(a, i, dy, dx, 2.0)
                c2(b, i, dy, dx, 2.0)
        p["b2"][b] = -1.0

    # Common front portals: bottom of vertical bridges and left
    # of horizontal bridges. Expand only across the wall thickness.
    for o in (26, 27):
        for dx in range(-1, 2):
            c2(o, 19, 0, dx)
        for dy in range(-1, 2):
            c2(o, 22, dy, 0)
    p["b2"][27] = -1.0

    # Alternate endpoint features remain available to axial tests.
    c2(28, 20)
    c2(29, 21)
    c2(30, 23)
    c2(31, 24)

    # ----------------------------------------------------------
    # Axial layer.
    # ----------------------------------------------------------
    f(0, 4)  # Goal source.

    for o, pool, wall, structural in (
        (1, "col", (16, 17), (20, 21)),
        (2, "row", (18, 19), (22, 23)),
    ):
        f(o, wall[0], pool=pool)
        f(o, wall[1], -1.0, pool)
        f(o, structural[0], pool=pool)
        f(o, structural[1], -1.0, pool)
        f(o, 26, -1.0)
        f(o, 27)
        f(o, 4, -1.0)
        p["b3"][o] = -1.0

    f(3, 1)  # Actual wall pixels.

    # Explicit player-normal features.
    for o, i in zip(range(4, 8), range(8, 12)):
        f(o, i)

    # A portal at the occupied platform's current front height.
    # The difference f3[8]-f3[9] is a binary seed.
    for o in (8, 9):
        f(o, 26)
        f(o, 27, -1.0)
        for i in (8, 9):
            f(o, i, pool="row")
        for i in (10, 11):
            f(o, i, pool="col")
    p["b3"][8] = -1.0
    p["b3"][9] = -2.0

    # Pressure-domain complement.
    p["b3"][12] = 1.0
    f(12, 0, -1.0)
    f(12, 2, -1.0)
    for i in range(8, 16):
        f(12, i, -1.0)

    f(13, 1, -1.0)
    p["b3"][13] = 1.0
    f(14, 2, -1.0)
    p["b3"][14] = 1.0

    source_pairs = ((15, 16), (17, 18), (19, 20), (21, 22))
    for (a, b), surface, player in zip(
        source_pairs, (12, 13, 14, 15), (8, 9, 10, 11)
    ):
        for o in (a, b):
            f(o, surface)
            f(o, player)
        p["b3"][b] = -1.0

    # Ready, player-aligned bridge crossing predicates.
    delta = 1.0 / 128.0
    bridge_pairs = ((23, 24), (25, 26), (27, 28), (29, 30))
    for (a, b), pool, coordinate, weighted, sign in (
        ((23, 24), "row", 30, 6, 1.0),
        ((25, 26), "row", 30, 6, -1.0),
        ((27, 28), "col", 31, 7, 1.0),
        ((29, 30), "col", 31, 7, -1.0),
    ):
        for o in (a, b):
            f(o, 3, 16.0)
            f(o, 5, 16.0, pool)
            f(o, coordinate, sign)
            f(o, weighted, -sign, pool)
        p["b3"][a] = -32.0 - delta
        p["b3"][b] = -32.0 - 2.0 * delta

    # ----------------------------------------------------------
    # Recurrent goal-route wave: h0-h1 is binary reachability,
    # and h2 is its accumulated arrival-time potential.
    # ----------------------------------------------------------
    for o in (0, 1):
        for dy, dx in ((0, 0),) + cardinal:
            r(o, 0, dy, dx)
            r(o, 1, dy, dx, -1.0)
        d(o, 0)
        for mask in (1, 2, 3):
            d(o, mask, -128.0)
    p["bh"][1] = -1.0

    r(2, 2)
    r(2, 0)
    r(2, 1, v=-1.0)

    # Portal-alignment wave. It is absent until the occupied
    # platform reaches a portal height.
    for o in (13, 14):
        for dy, dx in neighbourhood:
            r(o, 13, dy, dx)
            r(o, 14, dy, dx, -1.0)
        d(o, 8)
        d(o, 9, -1.0)
    p["bh"][14] = -1.0

    gamma = 1.06
    amplitude = (gamma - 1.0) / math.pow(gamma, 128)
    r(15, 15, v=gamma)
    r(15, 13, v=amplitude)
    r(15, 14, v=-amplitude)

    # ----------------------------------------------------------
    # Gated signed surface gradients and bridge gradients.
    # ----------------------------------------------------------
    B = 2048.0
    alignment_strength = 400.0

    gradient_specs = (
        (3, source_pairs[0], 4, (-1, 0), (1, 0), (29, 30)),
        (4, source_pairs[1], 5, (1, 0), (-1, 0), (27, 28)),
        (5, source_pairs[2], 6, (0, -1), (0, 1), (25, 26)),
        (6, source_pairs[3], 7, (0, 1), (0, -1), (23, 24)),
    )
    for o, (a, b), player, toward, away, bridge in gradient_specs:
        r(o, 2, *toward)
        r(o, 2, *away, v=-1.0)
        r(o, 15, *toward, v=alignment_strength)
        r(o, 15, *away, v=-alignment_strength)

        d(o, a, 2.0 * B)
        d(o, b, -2.0 * B)
        d(o, player, 2.0 * B)
        r(o, 13, v=2.0 * B)
        r(o, 14, v=-2.0 * B)

        ba, bb = bridge
        d(o, ba, B / delta)
        d(o, bb, -B / delta)
        p["bh"][o] = -3.0 * B

    # Same-delay neutral offset, independently computed.
    for a, b in source_pairs:
        d(12, a, 2.0 * B)
        d(12, b, -2.0 * B)
    for player in range(4, 8):
        d(12, player, 2.0 * B)
    r(12, 13, v=2.0 * B)
    r(12, 14, v=-2.0 * B)
    p["bh"][12] = -3.0 * B

    # Rectified positive/negative pressure fields.
    closed = 1.0e6
    for dy, dx in cardinal:
        r(7, 7, dy, dx, 0.25)
        r(8, 8, dy, dx, 0.25)
    for i in range(3, 7):
        r(7, i)
        r(8, i, v=-1.0)
    r(7, 12, v=-1.0)
    r(8, 12)
    d(7, 12, -closed)
    d(8, 12, -closed)

    # Pressure interpolation through separating walls.
    for o, main, wall in ((9, 7, 9), (10, 8, 10)):
        for dy, dx in cardinal:
            r(o, main, dy, dx, 0.25)
            r(o, wall, dy, dx, 0.25)
        d(o, 13, -closed)

    # Positive pressure contrast scores a size-transfer button.
    r(11, 7, v=4.0)
    r(11, 8, v=-4.0)
    for dy, dx in cardinal:
        r(11, 7, dy, dx, -1.0)
        r(11, 8, dy, dx, 1.0)
        r(11, 9, dy, dx, -1.0)
        r(11, 10, dy, dx, 1.0)
    d(11, 14, -closed)

    # ----------------------------------------------------------
    # Click head and action head.
    # ----------------------------------------------------------
    wc = p["Wc"][0]
    wc[32 + 11] = 1.0
    wc[32 + 7] = 0.001
    wc[32 + 8] = 0.001

    for i in range(3, 7):
        wc[32 + i] = 100.0
    wc[32 + 12] = -100.0

    exclusion = 1.0e6
    wc[14] = -exclusion
    for a, b in bridge_pairs:
        wc[a] += (exclusion - 0.125) / delta
        wc[b] -= (exclusion - 0.125) / delta

    p["bc"][0] = 1.0e-8
    p["bo"][:] = -10.0
    p["bo"][6] = 10.0

    # Global features and memory are intentionally unused.
    return p
