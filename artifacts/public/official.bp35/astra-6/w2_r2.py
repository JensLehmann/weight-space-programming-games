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

    def k1(o, c, dy=0, dx=0, value=1.0):
        p["W1"][o, c, dy + 2, dx + 2] += value

    def k2(o, c, dy=0, dx=0, value=1.0):
        p["W2"][o, c, dy + 1, dx + 1] += value

    def kh(o, c, dy=0, dx=0, value=1.0):
        p["Wh"][o, c, dy + 1, dx + 1] += value

    def f3(o, c, value=1.0, axis=0):
        p["W3"][o, c + 32 * axis] += value

    # ---------------------------------------------------------------
    # First convolution: terrain and sprite primitives.
    # ---------------------------------------------------------------

    # Hard material. Remove outlines close to a player, goal, or
    # destructible sprite.
    for colour in (5, 3, 8, 0, 11):
        k1(0, colour)
    for colour in (9, 7, 14, 12, 15):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                k1(0, colour, dy, dx, -1.0)

    # Absence of a nearby supporting surface. Horizontal dilation lets
    # a horizontal scan reach the centre of the first unsupported tile.
    for out, dy in ((1, -2), (2, 2)):
        p["b1"][out] = 1.0
        for dx in range(-2, 3):
            for colour in (5, 3, 8, 14, 15):
                k1(out, colour, dy, dx, -1.0)

    # Spike eyes, extended vertically to include the spike's centre.
    for dy in range(-2, 3):
        k1(3, 0, dy, 0)
        k1(3, 11, dy, -1)
        k1(3, 11, dy, 1)
    p["b1"][3] = -2.0

    # Ordinary interactive square centres: pink, solid blue, yellow.
    for colour in (14, 12, 15):
        k1(4, colour)
    for dy in (-2, 2):
        for dx in (-2, 2):
            k1(4, 3, dy, dx)
    p["b1"][4] = -4.0

    # Yellow and green square centres.
    for out, colour in ((6, 15), (7, 8)):
        k1(out, colour)
        for dy in (-2, 2):
            for dx in (-2, 2):
                k1(out, 3, dy, dx)
        p["b1"][out] = -4.0

    k1(8, 7)

    # Unique player anchor at sprite origin + (3, 3).
    for dy in (-2, -1, 0, 1):
        k1(9, 9, dy, 0)
    p["b1"][9] = -3.0

    k1(10, 16)

    # Coordinate hinges. Their alternating sum is a triangular wave
    # with peaks precisely at columns 3, 9, 15, ..., 63.
    for j in range(21):
        k1(11 + j, 17, value=63.0)
        p["b1"][11 + j] = -3.0 * j

    # ---------------------------------------------------------------
    # Second convolution.
    # ---------------------------------------------------------------

    k2(0, 0)
    k2(1, 1, -1, 0)
    k2(2, 2, 1, 0)
    k2(3, 8)
    k2(5, 4)
    k2(6, 7)
    k2(7, 6)
    k2(8, 9)
    k2(9, 10)
    k2(10, 11, value=1.0 / 63.0)
    p["b2"][11] = 1.0

    # Player-gated coordinates.
    k2(12, 11, value=1.0 / 63.0)
    k2(12, 9)
    p["b2"][12] = -1.0

    k2(13, 10)
    k2(13, 9)
    p["b2"][13] = -1.0

    # Unit peaks at tile-centre columns, zero at other integer columns.
    k2(15, 11)
    for j in range(1, 21):
        k2(15, 11 + j, value=-2.0 if j % 2 else 2.0)
    p["b2"][15] = -2.0

    k2(17, 3)

    # Hazards above and below the camera's tracking midpoint.
    k2(20, 3)
    k2(20, 10)
    p["b2"][20] = -1.0 - 34.0 / 63.0

    k2(21, 3)
    k2(21, 10, value=-1.0)
    p["b2"][21] = -1.0 + 34.0 / 63.0

    # ---------------------------------------------------------------
    # Axial features.
    # ---------------------------------------------------------------

    for out, src in (
        (0, 0), (1, 1), (2, 2), (3, 3),
        (7, 5), (8, 6), (9, 7),
        (10, 9), (11, 11), (27, 17),
    ):
        f3(out, src)

    # Non-centre penalty with a generous numerical margin.
    f3(4, 15, -2.0)
    p["b3"][4] = 1.0

    # Distances to horizontal destination anchors.
    f3(12, 12, axis=1)
    f3(12, 10, -1.0)
    p["b3"][12] = -6.0 / 63.0

    f3(13, 10)
    f3(13, 12, -1.0, axis=1)
    p["b3"][13] = 6.0 / 63.0

    f3(14, 12, axis=1)
    f3(14, 10, -1.0)
    p["b3"][14] = 6.0 / 63.0

    f3(15, 10)
    f3(15, 12, -1.0, axis=1)
    p["b3"][15] = -6.0 / 63.0

    f3(16, 8, axis=1)
    f3(17, 8, axis=2)

    # Signed displacements on the player's row/column cross.
    f3(18, 10)
    f3(18, 12, -1.0, axis=1)
    f3(18, 12, -1.0, axis=2)

    f3(19, 10, -1.0)
    f3(19, 12, axis=1)
    f3(19, 12, axis=2)

    f3(20, 9)
    f3(20, 13, -1.0, axis=1)
    f3(20, 13, -1.0, axis=2)

    f3(21, 9, -1.0)
    f3(21, 13, axis=1)
    f3(21, 13, axis=2)

    # Interactive centres on the cross, excluding the player itself.
    f3(22, 5)
    f3(22, 8, axis=1)
    f3(22, 8, axis=2)
    f3(22, 8, -1.0)
    p["b3"][22] = -1.0

    # Gravity inferred from the tracked player row.
    for axis in (1, 2):
        f3(23, 13, axis=axis)
        f3(23, 8, -34.0 / 63.0, axis=axis)
        f3(24, 13, -1.0, axis=axis)
        f3(24, 8, 34.0 / 63.0, axis=axis)

    # Opposite-side hazards in the player's column.
    f3(25, 20)
    f3(25, 8, axis=2)
    p["b3"][25] = -1.0

    f3(26, 21)
    f3(26, 8, axis=2)
    p["b3"][26] = -1.0

    # Yellow growth tie-breaker.
    f3(30, 7)
    f3(30, 10)
    p["b3"][30] = -1.0

    # Exclude the action-counter strip.
    f3(31, 9, 63.0)
    p["b3"][31] = -61.0

    # ---------------------------------------------------------------
    # Recurrent directional scans.
    #
    # Each gravity direction has five channels:
    #   S: vertical clearance, with extra reward for the goal
    #   D: danger propagated from spikes
    #   Q: safe clearance, sampled only at grid-centre columns
    #   L/R: horizontal maximum-envelope residuals
    # ---------------------------------------------------------------

    barrier = 2048.0

    for base, dy, unsupported, opposite_q in (
        (0, -1, 1, 7),
        (5, 1, 2, 2),
    ):
        scan, danger, quality, left, right = range(base, base + 5)

        kh(scan, scan, dy, 0)
        p["bh"][scan] = 1.0
        p["Uh"][scan, 0] = -barrier
        p["Uh"][scan, 9] = -barrier
        p["Uh"][scan, 31] = -barrier
        p["Uh"][scan, 3] = 40.0

        kh(danger, danger, dy, 0)
        p["Uh"][danger, 27] = 1.0
        p["Uh"][danger, 0] = -barrier
        p["Uh"][danger, 9] = -barrier
        p["Uh"][danger, 31] = -barrier

        kh(quality, scan)
        kh(quality, danger, value=-barrier)
        kh(quality, opposite_q, value=0.025)
        p["Uh"][quality, 4] = -barrier

        for out, dx in ((left, -1), (right, 1)):
            kh(out, out, 0, dx)
            kh(out, quality, 0, dx)
            kh(out, quality, value=-1.0)
            p["Uh"][out, unsupported] = -barrier
            p["Uh"][out, 0] = -barrier
            p["Uh"][out, 7] = -barrier
            p["Uh"][out, 27] = -barrier
            p["Uh"][out, 31] = -barrier
            p["bh"][out] = -0.03

    # Horizontal action probes.
    for out, quality, envelope, lo, hi, wrong in (
        (10, 2, 3, 12, 13, 24),
        (11, 2, 4, 14, 15, 24),
        (12, 7, 8, 12, 13, 23),
        (13, 7, 9, 14, 15, 23),
    ):
        kh(out, quality)
        kh(out, envelope)
        p["Uh"][out, lo] = -63.0 * 512.0
        p["Uh"][out, hi] = -63.0 * 512.0
        p["Uh"][out, 16] = barrier
        p["Uh"][out, 7] = -barrier
        p["Uh"][out, 0] = -barrier
        p["Uh"][out, 27] = -barrier
        p["Uh"][out, wrong] = -100000.0
        p["bh"][out] = -barrier

    # Adjacent click proposals. The vertical bonus compensates for
    # measuring clearance six pixels ahead of the player.
    for out, quality, danger, wrong, behind in (
        (14, 2, 1, 24, 20),
        (15, 7, 6, 23, 21),
    ):
        kh(out, quality)
        kh(out, danger, value=-barrier)
        p["Uh"][out, 22] = barrier
        p["Uh"][out, 17] = 7.0
        for feature in (18, 19, 20, 21):
            p["Uh"][out, feature] = -64.0 * 63.0
        p["Uh"][out, behind] -= 1024.0 * 63.0
        p["Uh"][out, wrong] = -100000.0
        p["bh"][out] = -barrier + 6.0 * 64.0 + 0.5

    # ---------------------------------------------------------------
    # Pooled action head.
    # ---------------------------------------------------------------

    def hg(unit, channel, weight=1.0):
        p["Wz"][unit, channel] += weight

    def fg(unit, channel, weight=1.0):
        p["Wz"][unit, 32 + channel] += weight

    for unit, channel in enumerate((10, 11, 12, 13)):
        hg(unit, channel)

    hg(4, 14)
    hg(4, 15)

    # Gravity switch fallback. Suppress it whenever a normal removal
    # proposal exists, keeping action selection and click targeting
    # consistent.
    for unit, wrong, hazard in ((5, 24, 25), (6, 23, 26)):
        fg(unit, 8, barrier)
        fg(unit, wrong, -100000.0)
        fg(unit, hazard, -barrier)
        hg(unit, 14, -512.0)
        hg(unit, 15, -512.0)
        p["bz"][unit] = -barrier + 5.0

    # Yellow growth fallback.
    fg(7, 9, barrier)
    fg(7, 8, -barrier)
    hg(7, 14, -512.0)
    hg(7, 15, -512.0)
    p["bz"][7] = -barrier + 4.5

    p["bo"][:] = -100.0
    p["bo"][3] = 0.0
    p["bo"][4] = 0.001
    p["bo"][6] = -0.01

    p["Wo"][3, 0] = 1.0
    p["Wo"][3, 2] = 1.0
    p["Wo"][4, 1] = 1.0
    p["Wo"][4, 3] = 1.0
    for unit in (4, 5, 6, 7):
        p["Wo"][6, unit] = 1.0

    # Small persistence term for otherwise equivalent routes.
    p["Wm"][0, 0] = 0.6
    p["Wma"][0, 3] = 1.0
    p["Wma"][0, 4] = -1.0
    p["Wo"][3, 128] = 0.005
    p["Wo"][4, 128] = -0.005

    # Spatial click readout.
    p["Wc"][0, 32 + 14] = 1.0
    p["Wc"][0, 32 + 15] = 1.0
    p["Wc"][0, 8] = 0.10
    p["Wc"][0, 9] = 0.05
    p["Wc"][0, 30] = 0.005
    p["Wc"][0, 27] = -1000.0

    return p
