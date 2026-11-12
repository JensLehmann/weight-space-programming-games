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
    p = {k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()}

    def c1(o, i, v=1.0, dy=0, dx=0):
        p["W1"][o, i, dy + 2, dx + 2] += v

    def c2(o, i, v=1.0, dy=0, dx=0):
        p["W2"][o, i, dy + 1, dx + 1] += v

    def feature(o, terms, bias=0.0):
        for i, v in terms:
            p["W3"][o, i] += v
        p["b3"][o] = bias

    def rec(o, i, v=1.0, dy=0, dx=0):
        p["Wh"][o, i, dy + 1, dx + 1] += v

    # First-layer player-centre detector.
    for colour, dy, dx in (
        (9, 0, 0), (11, 0, -1),
        (11, 0, 1), (5, 2, 0),
    ):
        c1(0, colour, dy=dy, dx=dx)
    p["b1"][0] = -2.0

    c1(1, 16, -1.0)
    p["b1"][1] = 0.5
    c1(2, 16)
    p["b1"][2] = -0.5
    c1(3, 16, 63.0)
    c1(4, 17)

    # Passable-looking pixels; morphology closes sprite outlines.
    for colour in (7, 8, 9, 10, 11, 12, 14, 15):
        c1(5, colour)

    for colour in (3, 5):
        c1(6, colour, dy=-2)
        c1(7, colour, dy=2)

    # Solid-coloured interiors exclude ghost-blue tiles.
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            for colour in (8, 12, 14, 15):
                c1(8, colour, dy=dy, dx=dx)
    p["b1"][8] = -8.0

    c1(9, 8)
    c1(10, 7)

    # Spike marker and orange-area dilation.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            c1(11, 0, dy=dy, dx=dx)
            c1(21, 15, dy=dy, dx=dx)

    # Player-body projections to movement decision locations.
    c1(12, 9, dx=2)
    c1(13, 9, dx=-2)

    # Adjacent-cell legality. Layer two extends these probes to
    # three pixels beyond the movement decision locations.
    for colour in (10, 7):
        c1(14, colour, dx=2)
        c1(15, colour, dx=-2)

    # At a ghost-blue centre the horizontal neighbour is background;
    # at a solid-blue centre it is also blue.
    c1(16, 12, dx=2)
    c1(16, 12, -1.0, dx=1)
    c1(17, 12, dx=-2)
    c1(17, 12, -1.0, dx=-1)

    c1(18, 16, -63.0)
    p["b1"][18] = 4.0
    c1(19, 16, 63.0)
    p["b1"][19] = -58.0
    c1(20, 16, 63.0)
    p["b1"][20] = -58.0

    # Column coordinate gated by the player-centre template.
    for colour, dy, dx in (
        (9, 0, 0), (11, 0, -1),
        (11, 0, 1), (5, 2, 0),
    ):
        c1(24, colour, 16.0, dy=dy, dx=dx)
    c1(24, 17)
    p["b1"][24] = -48.0

    # Second layer: bounded morphology and action geometry.
    c2(0, 0)
    c2(1, 0)
    c2(1, 1, -100.0)
    c2(2, 0)
    c2(2, 2, -100.0)

    for dy in range(-1, 2):
        for dx in range(-1, 2):
            c2(3, 5, dy=dy, dx=dx)
            c2(4, 5, dy=dy, dx=dx)
            c2(5, 8, dy=dy, dx=dx)
            c2(6, 8, dy=dy, dx=dx)
            c2(12, 11, dy=dy, dx=dx)
            c2(13, 11, dy=dy, dx=dx)
            c2(25, 21, dy=dy, dx=dx)
            c2(26, 21, dy=dy, dx=dx)

    p["b2"][4] = -1.0
    p["b2"][6] = -1.0
    p["b2"][13] = -1.0
    p["b2"][26] = -1.0

    # Budget-strip black pixels are not spike markers.
    c2(12, 20, -1000.0)
    c2(13, 20, -1000.0)

    c2(7, 6, dy=-1)
    c2(8, 7, dy=1)
    c2(9, 8)
    c2(10, 9)
    c2(11, 10)

    c2(14, 12, dx=1)
    c2(15, 13, dx=-1)

    c2(16, 3)
    c2(17, 18)
    c2(18, 19)

    c2(19, 3, -1.0)
    p["b2"][19] = 30.0
    c2(20, 3)
    p["b2"][20] = -36.0
    c2(21, 3, -1.0)
    p["b2"][21] = 32.0
    c2(22, 3)
    p["b2"][22] = -38.0

    c2(23, 3)
    p["b2"][23] = -61.0

    c2(24, 24)
    c2(27, 14, dx=1)
    c2(27, 16, dx=1)
    c2(28, 15, dx=-1)
    c2(28, 17, dx=-1)
    c2(29, 4)

    # Axial features.
    feature(0, [(3, 1), (4, -1), (23, -100)])
    feature(1, [(5, 1), (6, -1)])
    feature(2, [(7, 1)])
    feature(3, [(8, 1)])

    feature(4, [
        (64 + 12, 1), (64 + 13, -1),
        (25, 1), (26, -1),
    ], -1.0)

    feature(5, [(11, 1), (23, -100)])
    feature(6, [(16, -1.0 / 63.0)], 1.0)
    feature(7, [(16, 1.0 / 63.0)])
    feature(8, [(17, 0.25)])
    feature(9, [(18, 0.2)])

    # Movement masks require both the player-row projection and a
    # legal adjacent tile. Solid obstructions cannot receive a vote.
    feature(10, [(14, 1), (32, 1), (28, 1)], -2.0)
    feature(11, [(15, 1), (32, 1), (27, 1)], -2.0)

    feature(12, [(64 + 1, 1)])
    feature(13, [(64 + 2, 1)])

    # Immediately supporting removable blocks, gravity-gated.
    feature(14, [
        (9, 1), (64, 1), (64 + 1, 1),
        (19, -100), (20, -100), (10, -100),
    ], -2.0)
    feature(15, [
        (9, 1), (64, 1), (64 + 2, 1),
        (21, -100), (22, -100), (10, -100),
    ], -2.0)

    feature(16, [(10, 1)])
    feature(17, [(0, 1)])

    # Horizontal distance on the player's row.
    feature(18, [
        (29, 1), (32 + 24, -1), (32, 16),
    ], -16.0)
    feature(19, [
        (29, -1), (32 + 24, 1), (32, 16),
    ], -16.0)

    # Removable horizontal obstructions, independently gravity-gated.
    feature(22, [(9, 1), (32 + 1, 1), (10, -100)], -1.0)
    feature(23, [(9, 1), (32 + 2, 1), (10, -100)], -1.0)
    feature(24, [(1, 1)])
    feature(25, [(2, 1)])

    # Two discounted directed continuation fields.
    # In each five-channel group:
    # A+B = horizontal maximum, D+E = max(horizontal, vertical).
    hard = 64.0
    gh = 0.965
    gv = 0.985
    coupling = 0.004

    for base, other, dy, support, progress, frontier in (
        (0, 5, -1, 2, 6, 8),
        (5, 0, 1, 3, 7, 9),
    ):
        A, B, C, D, E = range(base, base + 5)

        for value in (D, E):
            rec(A, value, gh, dx=-1)
            rec(A, value, -gh, dx=1)
            rec(B, value, gh, dx=1)
            rec(C, value, gv, dy=dy)

        rec(C, other + 3, coupling)
        rec(C, other + 4, coupling)

        for out in (A, B):
            p["Uh"][out, 0] = hard
            p["Uh"][out, support] = hard
            p["Uh"][out, 4] = -hard
            p["Uh"][out, 16] = -hard
            p["bh"][out] = -2.0 * hard
        p["Uh"][B, 1] = -0.0015

        p["Uh"][C, 0] = hard
        p["Uh"][C, 4] = -hard
        p["Uh"][C, 16] = -hard
        p["Uh"][C, 1] = -0.0015
        p["Uh"][C, progress] = 0.0002
        p["Uh"][C, frontier] = 0.22
        p["Uh"][C, 5] = 2.0
        p["bh"][C] = -hard

        rec(D, A)
        rec(D, B)
        rec(D, C, -1.0)
        rec(E, C)

    # Action readouts.
    for base, heads, dy in (
        (0, (10, 11, 12), -1),
        (5, (13, 14, 15), 1),
    ):
        D, E = base + 3, base + 4
        for out, mask, dx in (
            (heads[0], 10, -1),
            (heads[1], 11, 1),
        ):
            rec(out, D, dx=dx)
            rec(out, E, dx=dx)
            p["Uh"][out, mask] = hard
            p["bh"][out] = -hard

        click_head = heads[2]
        rec(click_head, D, dy=dy)
        rec(click_head, E, dy=dy)
        if base == 0:
            masks = (14, 22, 25)
        else:
            masks = (15, 23, 24)
        for mask in masks:
            p["Uh"][click_head, mask] = hard
        p["Uh"][click_head, 18] = -0.15
        p["Uh"][click_head, 19] = -0.15
        p["Uh"][click_head, 4] = -hard
        p["bh"][click_head] = -hard

    # Pooled action-selection units.
    gate = 32.0
    up = 32 + 12
    down = 32 + 13
    orange = 32 + 16

    for z, head, flag in (
        (0, 10, up), (1, 11, up), (2, 12, up),
        (3, 13, down), (4, 14, down), (5, 15, down),
    ):
        p["Wz"][z, head] = 1.0
        p["Wz"][z, flag] = gate
        p["bz"][z] = -gate

    # Switch candidates compare continuation at the current player
    # position in reversed gravity against normal action values.
    for z, alternate, normal, flag in (
        (6, 15, (10, 11, 12), up),
        (7, 12, (13, 14, 15), down),
    ):
        p["Wz"][z, alternate] = 1.0
        for head in normal:
            p["Wz"][z, head] -= 2.0
        p["Wz"][z, orange] = gate
        p["Wz"][z, flag] = gate
        p["bz"][z] = -2.0 * gate - 0.005

    p["Wz"][8, orange] = 0.00001

    p["bo"][:] = -10.0
    p["bo"][3] = 0.0
    p["bo"][4] = 0.0
    p["bo"][6] = 0.0

    p["Wo"][3, 0] = 1.0
    p["Wo"][3, 3] = 1.0
    p["Wo"][4, 1] = 1.0
    p["Wo"][4, 4] = 1.0

    p["Wo"][6, 2] = 1.035
    p["Wo"][6, 5] = 1.035
    p["Wo"][6, 6] = 1.0
    p["Wo"][6, 7] = 1.0
    p["Wo"][6, 8] = 1.0

    # Very small tie-breaking persistence, subordinate to legality
    # and gravity-switch continuation.
    p["Wma"][0, 3] = 2.0
    p["Wma"][1, 4] = 2.0
    p["Wo"][3, 128] = 0.000003
    p["Wo"][4, 129] = 0.000003

    # Spatial click selection. Player-centre samples used for switch
    # evaluation must never themselves become click targets.
    p["Wc"][0, 32 + 12] = 0.30
    p["Wc"][0, 32 + 15] = 0.30
    p["Wc"][0, 16] = 0.015
    p["Wc"][0, 17] = -100.0
    p["Wc"][0, 4] = -100.0

    return p
