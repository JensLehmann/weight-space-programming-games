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
    p = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in shapes.items()
    }

    def c1(out, inp, dy=0, dx=0, weight=1.0):
        p["W1"][out, inp, dy + 2, dx + 2] += weight

    def c2(out, inp, dy=0, dx=0, weight=1.0):
        p["W2"][out, inp, dy + 1, dx + 1] += weight

    def rec(out, inp, dy=0, dx=0, weight=1.0):
        p["Wh"][out, inp, dy + 1, dx + 1] += weight

    def drive(out, inp, weight=1.0):
        p["Uh"][out, inp] += weight

    def local(out, inp, weight=1.0):
        p["W3"][out, inp] += weight

    def rows(out, inp, weight=1.0):
        p["W3"][out, 32 + inp] += weight

    def cols(out, inp, weight=1.0):
        p["W3"][out, 64 + inp] += weight

    colors = (10, 6, 15, 11, 12, 8)
    ghost_colors = (7, 14, 13)

    # ------------------------------------------------------------
    # First convolution: palette, coordinates, pad signatures,
    # fruit prefix counts, and local ghost presence.
    # ------------------------------------------------------------
    for tier, color in enumerate(colors):
        c1(0, color, weight=6 - tier)
        c1(1, color)
        c1(2, color, weight=tier)

    c1(3, 16, weight=-63.0)
    p["b1"][3] = 9.5

    c1(4, 16, weight=63.0)
    p["b1"][4] = -62.5

    c1(5, 16, weight=63.0)
    c1(6, 17, weight=63.0)

    c1(7, 16, weight=63.0)
    p["b1"][7] = -9.5

    c1(8, 17, weight=-63.0)
    p["b1"][8] = 27.5

    corners = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    for i, (sy, sx) in enumerate(corners):
        out = 9 + i
        c1(out, 9, 2 * sy, 0)
        c1(out, 9, 0, 2 * sx)
        c1(out, 5, 2 * sy, 2 * sx)
        c1(out, 5, 2 * sy, sx)
        c1(out, 5, sy, 2 * sx)
        p["b1"][out] = -4.0

    for tier in range(6):
        p["b1"][21 + tier] = 1.0
        for color in colors[:tier + 1]:
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    c1(13 + tier, color, dy, dx)
                    c1(21 + tier, color, dy, dx, -1.0)

    for color in ghost_colors:
        c1(19, color)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                c1(27, color, dy, dx)
                c1(28, color, dy, dx)
    p["b1"][28] = -1.0

    c1(20, 2)

    # ------------------------------------------------------------
    # Second convolution.
    # ------------------------------------------------------------
    c2(0, 0)
    c2(0, 3, weight=-64.0)

    c2(1, 1)
    c2(1, 3, weight=-4.0)

    for out, dy in ((2, 0), (3, -1)):
        c2(out, 2, dy, 0)
        c2(out, 7, dy, 0, -32.0)
        c2(out, 8, dy, 0, -32.0)

    for out, dx in ((4, -1), (5, 1)):
        c2(out, 0, 0, dx)
        c2(out, 3, 0, dx, -64.0)

    # Coordinate-coded pad markers and shifted copies.
    for out, dy, dx, coordinate in (
        (6, 0, 0, 6),
        (7, 0, -1, 6),
        (8, 0, 0, 5),
        (9, -1, 0, 5),
    ):
        for channel in range(9, 13):
            c2(out, channel, dy, dx, 64.0)
        c2(out, coordinate, dy, dx, -1.0)

    c2(10, 5)
    c2(11, 6)
    c2(12, 3)
    c2(12, 4)

    c2(13, 20)
    c2(13, 7, weight=-4.0)
    c2(13, 8, weight=-4.0)

    # Absence of each fruit prefix in a seven-by-seven window.
    for tier in range(6):
        out = 14 + tier
        p["b2"][out] = 1.0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                c2(out, 13 + tier, dy, dx, -1.0)

    c2(20, 19)
    c2(20, 3, weight=-4.0)

    # Rank of a locally endangered fruit.
    # Ghost presence is exactly count - ReLU(count - 1).
    c2(21, 27, weight=6.0)
    c2(21, 28, weight=-6.0)
    for channel in range(21, 27):
        c2(21, channel, weight=-1.0)
    c2(21, 3, weight=-64.0)

    # Absence of ghosts in a seven-by-seven window.
    p["b2"][22] = 1.0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            c2(22, 27, dy, dx, -1.0)

    # ------------------------------------------------------------
    # Axial feature layer.
    # ------------------------------------------------------------
    local(0, 0)
    local(1, 1)
    rows(2, 2)
    rows(3, 3)

    cols(4, 0)
    cols(5, 4)
    cols(6, 5)

    cols(7, 6)
    cols(8, 7)
    rows(9, 8)
    rows(10, 9)

    local(11, 10)
    local(12, 11)
    local(13, 12)
    local(14, 13)

    p["b3"][15] = 6.0
    for channel in range(14, 20):
        local(15, channel, -1.0)

    local(16, 20)

    # Their sum is max(row danger rank, column danger rank).
    rows(17, 21)
    cols(17, 21, -1.0)
    cols(18, 21)

    p["b3"][19] = 1.0
    local(19, 22, -1.0)

    # Boundary costs for ghost deflection.
    p["b3"][20] = 6.0
    local(20, 11, -1.0)

    p["b3"][21] = -57.0
    local(21, 11)

    p["b3"][22] = 16.0
    local(22, 10, -1.0)

    p["b3"][23] = -57.0
    local(23, 10)

    # ------------------------------------------------------------
    # Recurrent rank and header broadcasts.
    # ------------------------------------------------------------

    # H0: header tier propagated downward.
    rec(0, 0, -1, 0)
    drive(0, 3)
    drive(0, 2, -1.0)

    # H1: left-to-right prefix maximum increment.
    rec(1, 1, 0, -1)
    drive(1, 5)
    drive(1, 4, -1.0)

    # H2: right-to-left maximum increment over that prefix.
    # Global rank = H1 + H2 + F3[4].
    rec(2, 2, 0, 1)
    rec(2, 1, 0, 1)
    rec(2, 1, 0, 0, -1.0)
    drive(2, 6)
    drive(2, 4, -1.0)

    def global_rank(out, coefficient):
        rec(out, 1, weight=coefficient)
        rec(out, 2, weight=coefficient)
        drive(out, 4, coefficient)

    def tier_difference(out, coefficient):
        # Requested tier minus smallest remaining tier.
        rec(out, 0, weight=coefficient)
        drive(out, 2, coefficient)
        global_rank(out, coefficient)
        p["bh"][out] -= 6.0 * coefficient

    # H3: uniform unit sources on lowest-tier fruit.
    drive(3, 0)
    global_rank(3, -1.0)
    drive(3, 1, 16.0)
    p["bh"][3] = -15.0

    # ------------------------------------------------------------
    # Damped diffusion fields.
    # ------------------------------------------------------------
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for dy, dx in neighbors:
        rec(4, 4, dy, dx, 0.20)
        rec(5, 5, dy, dx, 0.245)
        rec(6, 6, dy, dx, 0.245)
        rec(11, 11, dy, dx, 0.245)

    rec(4, 3)
    rec(5, 3)

    # Larger fruit act as obstacles during merging.
    drive(6, 1)
    rec(6, 3, weight=-1.0)

    drive(11, 16)

    # ------------------------------------------------------------
    # Pad-coordinate potentials.
    # ------------------------------------------------------------
    rec(7, 7, 0, -1)
    drive(7, 8)
    drive(7, 7, -1.0)

    rec(8, 8, -1, 0)
    drive(8, 10)
    drive(8, 9, -1.0)

    rec(9, 7)
    drive(9, 7)
    drive(9, 12)
    p["bh"][9] = -64.0

    rec(10, 8)
    drive(10, 9)
    drive(10, 11)
    p["bh"][10] = -64.0

    def goal_potential(out, coefficient):
        drive(out, 11, coefficient)
        drive(out, 12, coefficient)
        rec(out, 9, weight=-2.0 * coefficient)
        rec(out, 10, weight=-2.0 * coefficient)

    # ------------------------------------------------------------
    # Behavioral heads.
    # ------------------------------------------------------------

    # H13: annular merging preference.
    rec(13, 5)
    rec(13, 4, weight=-12.0)
    rec(13, 6, weight=-1.5)
    rec(13, 11, weight=-1.0)
    tier_difference(13, 4.0)
    p["bh"][13] -= 2.0

    # H14 is zero precisely in the integer-valued merging regime.
    # Otherwise it is at least one half.
    p["bh"][14] = 0.5
    tier_difference(14, -1.0)

    # H12: deflect an endangered ghost during merging.
    danger_scale = 65536.0
    p["bh"][12] = 20000.0 - 6.0 * danger_scale

    drive(12, 17, danger_scale)
    drive(12, 18, danger_scale)
    rec(12, 0, weight=danger_scale)
    drive(12, 2, danger_scale)

    # Require a nearby ghost and an active merging regime.
    drive(12, 19, 1.0e7)
    p["bh"][12] -= 1.0e7
    rec(12, 14, weight=-1.0e7)

    # Choose the side of the ghost farther from vulnerable fruit.
    rec(12, 5, weight=-256.0)

    # Do not repeatedly pull a boundary-clamped ghost outward.
    for channel in range(20, 24):
        drive(12, channel, -1024.0)

    # H15: exact-tier delivery.
    #
    # For integer d:
    #   K - 1.5*K*d - 2*K*ReLU(0.5-d)
    # is zero at d=0 and at most -K/2 otherwise.
    mode_scale = 2048.0
    p["bh"][15] = 512.0 + mode_scale
    goal_potential(15, 1.0)
    tier_difference(15, -1.5 * mode_scale)
    rec(15, 14, weight=-2.0 * mode_scale)
    rec(15, 11, weight=-8.0)

    rank_gate = 16384.0
    drive(15, 15, rank_gate)
    global_rank(15, -rank_gate)

    # ------------------------------------------------------------
    # Readout.
    # ------------------------------------------------------------
    p["Wc"][0, 32 + 12] = 1.0
    p["Wc"][0, 32 + 13] = 1.0
    p["Wc"][0, 32 + 15] = 1.0

    epsilon = 1.0e-5
    p["Wc"][0, 11] = epsilon
    p["Wc"][0, 12] = epsilon
    p["Wc"][0, 32 + 9] = -2.0 * epsilon
    p["Wc"][0, 32 + 10] = -2.0 * epsilon

    # When all behavioral heads are inactive, wait away from
    # fruit and ghosts so intentional downgrades can occur.
    p["Wc"][0, 32 + 5] = -0.001
    p["Wc"][0, 32 + 11] = -0.001

    p["Wc"][0, 13] = -1.0e6

    p["bo"][:] = -20.0
    p["bo"][6] = 1.0
    p["bo"][0] = -5.0

    # Reset on the explicit infeasible-objective indication.
    p["Wz"][0, 32 + 14] = 1.0
    p["Wo"][0, 0] = 10.0

    return p
