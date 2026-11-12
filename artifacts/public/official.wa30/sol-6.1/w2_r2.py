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
    p = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in shapes.items()
    }

    # Direction order: up, down, left, right.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    opposite = [1, 0, 3, 2]

    def first(out, color, dr=0, dc=0, weight=1.0):
        p["W1"][out, color, dr + 2, dc + 2] += weight

    def second(out, inp, dr=0, dc=0, weight=1.0):
        p["W2"][out, inp, dr + 1, dc + 1] += weight

    def feature_max(channel):
        return 32 + channel

    # Facing-edge markers and attached-crate markers.
    for i, (fr, fc) in enumerate(directions):
        br, bc = -fr, -fc
        lr, lc = (0, 1) if fr else (1, 0)

        first(i, 0)
        for dr, dc in [
            (br, bc),
            (2 * br, 2 * bc),
            (br + lr, bc + lc),
            (br - lr, bc - lc),
        ]:
            first(i, 14, dr, dc)
        first(i, 14, br - 2 * lr, bc - 2 * lc, -1.0)
        p["b1"][i] = -4.0

        c = 4 + i
        first(c, 0)
        first(c, 14, br, bc)
        first(c, 0, fr, fc)
        first(c, 9, 2 * fr, 2 * fc)
        first(c, 9, 2 * fr + lr, 2 * fc + lc)
        p["b1"][c] = -4.0

    core = [(0, 0), (0, 1), (1, 0), (1, 1)]

    # Crate collection source. Helper-held crates receive a weaker source.
    for dr, dc in core:
        first(8, 9, dr, dc)
    for border in (4, 3):
        first(8, border, -1, 0)
        first(8, border, 0, -1)
    first(8, 5, -1, 0, 0.625)
    first(8, 5, 0, -1, 0.625)
    p["b1"][8] = -5.0

    first(9, 15)
    first(9, 11)

    # Displaced occupancy masks for both ends of a carrying footprint.
    # Constant-plane subtraction also identifies image padding.
    for i, (dr, dc) in enumerate(directions):
        c = 10 + i
        for rr in (-1, 0, 1):
            for cc in (-1, 0, 1):
                r, col = dr + rr, dc + cc
                for color in (5, 12, 4, 3, 15, 11):
                    first(c, color, r, col)
                first(c, 18, r, col, -1.0)
        p["b1"][c] = 9.0

    # Exact patterned-tile motifs. The motifs distinguish the barriers
    # from uniform colour-2 staging regions and their exposed edges.
    for i, (dr, dc) in enumerate(directions):
        c = 14 + i
        r, col = 2 * dr, 2 * dc

        # Up/left use the lower-right interior motif; down/right use
        # the upper-left interior motif, keeping all tests inside 5x5.
        sign = 1 if i in (0, 2) else -1
        first(c, 2, r, col)
        first(c, 1, r + sign, col)
        first(c, 1, r, col + sign)
        first(c, 2, r + sign, col + sign)
        p["b1"][c] = -3.0

    # Goal-interior sources shifted toward the crate end.
    for i, (dr, dc) in enumerate(directions):
        c = 18 + i
        r, col = 2 * dr, 2 * dc
        first(c, 2, r, col, 4.0)
        for nr, nc in directions:
            rr, cc = r + nr, col + nc
            if -2 <= rr <= 2 and -2 <= cc <= 2:
                first(c, 9, rr, cc)
        p["b1"][c] = -4.0

    # Player-carried crate core, for goal-entry prediction.
    for dr, dc in core:
        first(22, 9, dr, dc)
    first(22, 0, -1, 0)
    first(22, 0, 0, -1)
    p["b1"][22] = -5.0

    # Gated player coordinates.
    first(24, 16)
    first(24, 14)
    p["b1"][24] = -1.0
    first(25, 17)
    first(25, 14)
    p["b1"][25] = -1.0

    # Three-pixel goal-border lines beyond the current crate cell.
    for i, (dr, dc) in enumerate(directions):
        c = 26 + i
        lr, lc = (0, 1) if dr else (1, 0)
        for t in (-1, 0, 1):
            first(c, 9, 2 * dr + t * lr,
                  2 * dc + t * lc)
        p["b1"][c] = -2.0

    first(30, 5)
    first(30, 12)

    # These highlights mean ACTION5 can grab or remove the facing object.
    first(31, 3)
    first(31, 11)

    # One unladen readout point at the player's interior, and one
    # carrying readout point at the midpoint of its two-cell footprint.
    for i, (dr, dc) in enumerate(directions):
        second(0, i, dr, dc)
        second(0, 4 + i, dr, dc, -1.0)

        if i == 0:
            mr, mc = 1, 0
        elif i == 2:
            mr, mc = 0, 1
        else:
            mr, mc = 0, 0

        second(1, 4 + i, mr, mc)
        second(27 + i, 4 + i, mr, mc)

    # Crates with visible goal continuation beside them are not targets.
    second(6, 8)
    for channel, dr, dc in [
        (18, 0, 0), (19, 1, 0),
        (20, 0, 0), (21, 0, 1),
    ]:
        second(6, channel, dr, dc, -1.0)

    second(7, 9)

    # Unladen wall mask.
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            second(8, 30, dr, dc)

    # Reconstruct the patterned tile perimeter from its exact motifs.
    for dc in (-1, 0, 1):
        second(8, 14, 1, dc)
        second(8, 15, -1, dc)
    for dr in (-1, 0, 1):
        second(8, 16, dr, 1)
        second(8, 17, dr, -1)

    # Four carrying geometries. Patterned barriers constrain the player
    # end, while the crate end can enter them for helper handoff.
    for i, (dr, dc) in enumerate(directions):
        j = opposite[i]
        jr, jc = directions[j]

        second(9 + i, 10 + i, dr, dc)
        second(9 + i, 10 + j, jr, jc)

        for rr in (-1, 0, 1):
            for cc in (-1, 0, 1):
                second(9 + i, 14 + j, rr, cc)

        second(13 + i, 18 + i)

        # Auxiliary handoff targets on patterned tiles.
        if i in (0, 2):
            second(13 + i, 14 + i, 1, 1, 0.25)
        else:
            second(13 + i, 14 + i, 0, 0, 0.25)

    second(17, 31)
    second(19, 24)
    second(20, 25)

    # At a carried crate core, test the next cell's visible goal border.
    for i, (dr, dc) in enumerate(directions):
        second(21 + i, 22)
        sr = 1 if dr > 0 else 0
        sc = 1 if dc > 0 else 0
        second(21 + i, 26 + i, sr, sc)
        p["b2"][21 + i] = -1.0

    # Local features pass through the axial block.
    for i in range(32):
        p["W3"][i, i] = 1.0

    # Suppress the three incorrect carrying geometries near their
    # directional readout sites using row/column broadcasts.
    for orientation in range(4):
        out = 2 + orientation
        for actual in range(4):
            if actual != orientation:
                c = 27 + actual
                p["W3"][out, 32 + c] = 1.0
                p["W3"][out, 64 + c] = 1.0

    barrier = 8192.0
    source_scale = 64.0

    # Six recurrent potential fields:
    # collection, enemies, and four carrying orientations.
    for channel in range(6):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    p["Wh"][channel, channel, dr + 1, dc + 1] = 0.125

    p["Uh"][0, 6] = source_scale
    p["Uh"][0, 8] = -barrier
    p["Uh"][1, 7] = source_scale
    p["Uh"][1, 8] = -barrier

    for i in range(4):
        field = 2 + i
        p["Uh"][field, 13 + i] = source_scale
        p["Uh"][field, 9 + i] = -barrier
        p["Uh"][field, 2 + i] = -barrier

    # Directional samples of the appropriate potentials.
    for i, (dr, dc) in enumerate(directions):
        unladen = 6 + i
        carrying = 10 + i

        p["Wh"][unladen, 0, dr + 1, dc + 1] = 1.0
        p["Wh"][unladen, 1, dr + 1, dc + 1] = 8.0
        p["Uh"][unladen, 0] = barrier
        p["bh"][unladen] = -barrier

        for field in range(2, 6):
            p["Wh"][carrying, field, dr + 1, dc + 1] = 1.0
        p["Uh"][carrying, 1] = barrier
        p["bh"][carrying] = -barrier

    # Direction scores, carrying flags, coordinates, and border forecasts.
    for i in range(4):
        p["Wz"][i, 6 + i] = 1.0
        p["Wz"][i, feature_max(1)] = -barrier
        p["Wz"][4 + i, 10 + i] = 1.0
        p["Wz"][12 + i, feature_max(21 + i)] = 1.0
        p["Wz"][21 + i, feature_max(27 + i)] = 1.0

    p["Wz"][16, feature_max(17)] = 1.0
    p["Wz"][18, feature_max(1)] = 1.0
    p["Wz"][19, feature_max(19)] = 1.0
    p["Wz"][20, feature_max(20)] = 1.0

    # Release when every carrying-direction signal is zero.
    p["Wz"][25, feature_max(1)] = 0.02
    for i in range(4):
        p["Wz"][25, 10 + i] = -1.0

    # Release if carrying while no goal/handoff source remains visible.
    p["Wz"][26, feature_max(1)] = 1.0
    for i in range(4):
        p["Wz"][26, feature_max(13 + i)] = -8.0

    sharp = 20.0

    # Forecast flags and confirmation by the action actually taken.
    for i in range(4):
        p["Wmz"][i, 12 + i] = sharp
        p["bm"][i] = -0.5 * sharp

        p["Wm"][4 + i, i] = sharp
        p["Wma"][4 + i, 1 + i] = 2.0 * sharp
        p["bm"][4 + i] = -2.0 * sharp

    # Holding state and immediately preceding release.
    p["Wmz"][8, 18] = sharp
    p["bm"][8] = -0.5 * sharp
    p["Wm"][9, 8] = sharp
    p["Wma"][9, 5] = 2.0 * sharp
    p["bm"][9] = -2.0 * sharp

    # Orientation memories and release-orientation flags.
    for i in range(4):
        p["Wmz"][10 + i, 21 + i] = sharp
        p["bm"][10 + i] = -0.5 * sharp

        p["Wm"][14 + i, 10 + i] = sharp
        p["Wma"][14 + i, 5] = 2.0 * sharp
        p["bm"][14 + i] = -2.0 * sharp

    # Approximately linear current and one-step-delayed coordinates.
    coordinate_scale = 0.05
    p["Wmz"][18, 19] = coordinate_scale
    p["Wmz"][19, 20] = coordinate_scale
    p["Wm"][26, 18] = 1.0
    p["Wm"][27, 19] = 1.0

    # Four inequalities for a stationary position, and four more for
    # returning to the position from two actions ago.
    gain = 10000.0
    tolerance = 0.001
    comparisons = [
        (20, 18, 19, 1.0), (21, 18, 19, -1.0),
        (22, 19, 20, 1.0), (23, 19, 20, -1.0),
        (28, 26, 19, 1.0), (29, 26, 19, -1.0),
        (30, 27, 20, 1.0), (31, 27, 20, -1.0),
    ]
    for out, old, current, sign in comparisons:
        p["Wm"][out, old] = -sign * gain
        p["Wmz"][out, current] = sign * gain * coordinate_scale
        p["bm"][out] = gain * tolerance

    # Carrying now, following a movement rather than a pickup/release.
    p["Wmz"][24, 18] = sharp
    for action in range(1, 5):
        p["Wma"][24, action] = sharp
    p["bm"][24] = -1.5 * sharp

    for i in range(4):
        action = 1 + i
        p["Wo"][action, i] = 1.0
        p["Wo"][action, 4 + i] = 1.0

    # ACTION5 priorities: highlighted object, confirmed goal entry,
    # no carrying path, or a blocked/cyclic carrying position.
    p["Wo"][5, 16] = 5000.0
    p["Wo"][5, 25] = 100000.0
    p["Wo"][5, 26] = 5000.0

    for i in range(4):
        p["Wo"][5, 128 + 4 + i] = 1000.0

    p["Wo"][5, 128 + 9] = -4000.0

    for unit in list(range(20, 24)) + list(range(28, 32)):
        p["Wo"][5, 128 + unit] = 300.0
    p["Wo"][5, 128 + 24] = 600.0
    p["bo"][5] = -2350.0

    # A perpendicular step after release changes the attachment geometry.
    for orientation in (14, 15):
        p["Wo"][4, 128 + orientation] += 250.0
    p["bo"][4] += 500.0

    for orientation in (16, 17):
        p["Wo"][1, 128 + orientation] += 250.0
    p["bo"][1] += 500.0

    p["bo"][0] = -1.0e6
    p["bo"][6] = -1.0e6
    p["bo"][7] = -1.0e6

    return p
