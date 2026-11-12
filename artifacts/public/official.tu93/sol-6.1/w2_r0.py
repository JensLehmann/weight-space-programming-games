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

    W1, b1 = p["W1"], p["b1"]
    W2, b2 = p["W2"], p["b2"]
    W3, b3 = p["W3"], p["b3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]
    Wz, bz = p["Wz"], p["bz"]
    Wo, bo = p["Wo"], p["bo"]

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    opposite = [1, 0, 3, 2]

    # ------------------------------------------------------------
    # Layer 1: sprite centers, coordinates, and local track edges.
    # ------------------------------------------------------------
    def tile_unit(out, colors, bias=-8.0):
        for color in colors:
            W1[out, color, 1:4, 1:4] = 1.0
        b1[out] = bias

    tile_unit(0, [9, 4])
    tile_unit(1, [0, 2, 4, 8, 9, 11, 12, 13, 14, 15])
    tile_unit(2, [14])

    W1[3, 16, 2, 2] = 63.0
    W1[4, 17, 2, 2] = 63.0

    for d, (dr, dc) in enumerate(directions):
        W1[5 + d, 2, 2 + 2 * dr, 2 + 2 * dc] = 1.0

    def heading_units(first, body_color, active_only=False):
        for d, (dr, dc) in enumerate(directions):
            out = first + d
            tile_unit(out, [body_color, 15, 11], bias=-9.0)
            markers = [11] if active_only else [15, 11]
            for marker in markers:
                W1[out, marker, 2 + dr, 2 + dc] += 1.0

    heading_units(9, 8)
    heading_units(13, 12)
    heading_units(17, 13, active_only=True)

    tile_unit(21, [0, 4, 8, 9, 11, 12, 13, 14, 15])
    tile_unit(22, [8, 15, 11])
    tile_unit(23, [13, 15, 11])

    W1[24, 6, 2, 2] = 1.0

    # ------------------------------------------------------------
    # Layer 2:
    # 0 track, 1 goal, 2 row, 3 column
    # 4..7 legal player directions
    # 8 guard, 9 follower
    # 10..11 hinges identifying patrols at junctions
    # 12..15 edge midpoint predicates
    # 16..19 guard / active-follower headings
    # 20..23 continuing patrol headings
    # 24..27 reversing patrol headings
    # 28 node centers, 29..30 player coordinates, 31 budget
    # ------------------------------------------------------------
    for out, src in [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (8, 22), (9, 23), (28, 21), (31, 24),
    ]:
        W2[out, src, 1, 1] = 1.0

    def gated_coordinate(out, coordinate, predicate):
        W2[out, coordinate, 1, 1] = 1.0
        W2[out, predicate, 1, 1] = 64.0
        b2[out] = -64.0

    gated_coordinate(29, 3, 0)
    gated_coordinate(30, 4, 0)

    for d, (dr, dc) in enumerate(directions):
        er, ec = 1 + dr, 1 + dc

        W2[12 + d, 5 + d, er, ec] = 1.0

        W2[4 + d, 0, 1, 1] = 1.0
        W2[4 + d, 5 + d, er, ec] = 1.0
        b2[4 + d] = -1.0

        W2[16 + d, 9 + d, 1, 1] = 1.0
        W2[16 + d, 17 + d, 1, 1] = 1.0

        W2[20 + d, 13 + d, 1, 1] = 1.0
        W2[20 + d, 5 + d, er, ec] = 1.0
        b2[20 + d] = -1.0

        W2[24 + d, 13 + d, 1, 1] = 1.0
        W2[24 + d, 5 + d, er, ec] = -1.0

    # Their difference is one precisely at a patrol center with
    # at least three incident edges, and zero elsewhere.
    for out, bias in [(10, -4.0), (11, -5.0)]:
        for d in range(4):
            W2[out, 13 + d, 1, 1] = 2.0
            dr, dc = directions[d]
            W2[out, 5 + d, 1 + dr, 1 + dc] = 1.0
        b2[out] = bias

    # ------------------------------------------------------------
    # Layer 3: graph features and destination selectors.
    # ------------------------------------------------------------
    W3[0, 0] = 1.0
    W3[1, 1] = 1.0
    W3[20, 28] = 1.0

    W3[2, 20:28] = 1.0

    for d in range(4):
        W3[3 + d, 16 + d] = 1.0
        W3[3 + d, 20 + d] = 1.0
        W3[3 + d, 24 + opposite[d]] = 1.0

    # Avoid ordinary leaves, but permit goals and guard captures.
    b3[7] = 2.0
    W3[7, 12:16] = -1.0
    W3[7, 1] = -2.0
    W3[7, 8] = -2.0

    # Three hinges make a tent exactly six pixels from the player.
    for d in range(4):
        axial_offset = 64 if d < 2 else 32
        coordinate = 2 if d < 2 else 3
        player_coordinate = 29 if d < 2 else 30
        sign = -1.0 if d in (0, 2) else 1.0

        for j, threshold in enumerate((5.0, 6.0, 7.0)):
            out = 8 + 3 * d + j
            W3[out, coordinate] = sign
            W3[out, axial_offset + player_coordinate] = -sign
            W3[out, axial_offset + 4 + d] = 128.0
            b3[out] = -128.0 - threshold

    for d in range(4):
        W3[21 + d, 4 + d] = 1.0

    W3[25, 29] = 1.0
    W3[26, 30] = 1.0
    W3[27, 31] = 1.0

    # Occlusion-risk sources: a junction patrol sharing a row or
    # column with a visible stationary guard. The patrol itself
    # may conceal another guard.
    for out, axial_offset in [(28, 32), (29, 64)]:
        W3[out, 10] = 1.0
        W3[out, 11] = -1.0
        W3[out, axial_offset + 8] = 1.0
        b3[out] = -1.0

    W3[30, 9] = 1.0
    W3[31, 8] = 1.0

    # ------------------------------------------------------------
    # Recurrent channels:
    # 0..1 goal flood / overflow
    # 2 integrated goal reachability
    # 3..6 action scores
    # 7..10 directional danger signals
    # 11..12 guard flood / overflow
    # 13 integrated guard reachability
    # 14 target-local clocks
    # 15 immediate node danger
    # ------------------------------------------------------------
    uncertainty_source = 750_000.0

    for d, (dr, dc) in enumerate(directions):
        channel = 7 + d
        Wh[channel, channel, 1 - dr, 1 - dc] = 0.1
        Uh[channel, 3 + d] = 1_000_000.0
        Uh[channel, 28] = uncertainty_source
        Uh[channel, 29] = uncertainty_source

    # Cancel signals at their sources: an enemy's current square
    # remains capturable. Retain hazards at neighboring nodes.
    for d in range(4):
        Wh[15, 7 + d, 1, 1] = 1.0
        Uh[15, 3 + d] = -1_000_000.0
    Uh[15, 28] = -4.0 * uncertainty_source
    Uh[15, 29] = -4.0 * uncertainty_source
    Uh[15, 20] = 2_000_000.0
    bh[15] = -2_000_000.5

    # Delayed seeding lets directional danger settle before either
    # reachability field starts propagating.
    Wh[14, 14, 1, 1] = 1.0
    Uh[14, 1] = 1.0
    Uh[14, 31] = 1.0

    def flood_pair(raw, overflow, goal_field):
        for dr, dc in directions:
            for out in (raw, overflow):
                Wh[out, raw, 1 + dr, 1 + dc] += 1.0
                Wh[out, overflow, 1 + dr, 1 + dc] -= 1.0

        for out in (raw, overflow):
            Uh[out, 0] = 5.0
            Wh[out, 14, 1, 1] = 1.0
            Wh[out, 15, 1, 1] = -1024.0

            if goal_field:
                Uh[out, 1] = -8.0
                Uh[out, 31] = -1024.0
                Uh[out, 2] = -1024.0
            else:
                Uh[out, 31] = -8.0
                Uh[out, 1] = -1024.0

        bh[raw] = -5.0
        bh[overflow] = -6.0

    flood_pair(0, 1, True)
    flood_pair(11, 12, False)

    Wh[2, 2, 1, 1] = 1.0
    Wh[2, 0, 1, 1] = 1.0
    Wh[2, 1, 1, 1] = -1.0

    Wh[13, 13, 1, 1] = 1.0
    Wh[13, 11, 1, 1] = 1.0
    Wh[13, 12, 1, 1] = -1.0

    for d in range(4):
        out = 3 + d
        Wh[out, 2, 1, 1] = 10.0
        Wh[out, 13, 1, 1] = 2.0
        Wh[out, 15, 1, 1] = -65536.0

        first = 8 + 3 * d
        Uh[out, first] = 4096.0
        Uh[out, first + 1] = -8192.0
        Uh[out, first + 2] = 4096.0
        Uh[out, 7] = -300.0
        bh[out] = -4032.0

    # ------------------------------------------------------------
    # Pooled action readout.
    # ------------------------------------------------------------
    for d in range(4):
        Wz[d, 3 + d] = 1.0
        Wo[1 + d, d] = 1.0

        Wz[5 + d, 32 + 21 + d] = 1.0
        Wo[1 + d, 5 + d] = 100.0

        Wz[9 + d, 3 + d] = -1.0
        bz[9 + d] = 1.0
        Wo[1 + d, 9 + d] = -10000.0

    Wz[4, 32 + 25] = 64.0
    Wz[4, 32 + 26] = 1.0

    # ------------------------------------------------------------
    # Memory: six recent positions and destination-equality tests.
    # ------------------------------------------------------------
    p["Wmz"][0, 4] = 1e-6
    for age in range(1, 6):
        p["Wm"][age, age - 1] = 1.0

    penalties = [12.0, 10.0, 8.0, 6.0, 4.0, 2.0]
    destination_offsets = [-384.0, 384.0, -6.0, 6.0]
    slope = 4.0

    for age, penalty in enumerate(penalties):
        for d, offset in enumerate(destination_offsets):
            first = 6 + 8 * age + 2 * d
            for j, shift in enumerate((0.5, -0.5)):
                unit = first + j
                p["Wmz"][unit, 4] = slope
                p["Wm"][unit, age] = -slope * 1e6
                p["bm"][unit] = slope * (offset + shift)

            Wo[1 + d, 128 + first] = -0.5 * penalty
            Wo[1 + d, 128 + first + 1] = 0.5 * penalty

    # Low-budget recovery, suppressed by strong destination scores.
    Wz[25, 64 + 27] = -640000.0
    Wz[25, 3:7] = -1.0
    bz[25] = 1600.0
    Wo[0, 25] = 10000.0

    bo[:] = -1_000_000.0
    bo[1:5] = 10000.0

    return p
