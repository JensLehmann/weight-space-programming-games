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

    def first(out, channel, dy=0, dx=0, weight=1.0):
        W1[out, channel, dy + 2, dx + 2] += weight

    def second(out, channel, dy=0, dx=0, weight=1.0):
        W2[out, channel, dy + 1, dx + 1] += weight

    # Direction order: up, down, left, right.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Six exact local signatures:
    # top endpoint, bottom endpoint, two upward corners,
    # left endpoint, right endpoint.
    signatures = [
        {1},
        {0},
        {0, 3},
        {0, 2},
        {3},
        {2},
    ]
    groups = [(0,), (1,), (2, 3), (4, 5)]

    # Piece signatures. At a cell centre, immediate neighbours
    # are rendering filler, and distance-two samples indicate
    # occupancy of the adjacent cells.
    for out, required in enumerate(signatures):
        for dy, dx in directions:
            first(out, 4, dy, dx, weight=5.0)
            first(out, 5, dy, dx, weight=5.0)

        for direction, (dy, dx) in enumerate(directions):
            weight = 1.0 if direction in required else -1.0
            first(out, 4, 2 * dy, 2 * dx, weight=weight)
            first(out, 5, 2 * dy, 2 * dx, weight=weight)

        b1[out] = -(20.0 + len(required) - 1.0)
        second(out, out)

    # Target-neighbour samples, extended to distance three
    # by the second convolution.
    for direction, (dy, dx) in enumerate(directions):
        first(6 + direction, 11, 2 * dy, 2 * dx)
    first(10, 11)

    # Selected vertical/horizontal axes.
    for offset in range(-2, 3):
        first(11, 0, dy=offset)
        first(12, 0, dx=offset)
    first(11, 10)
    first(12, 10)
    b1[11] = -2.0
    b1[12] = -2.0

    # Coordinates and their complements.
    first(13, 16)
    first(14, 17)
    first(15, 18)
    first(15, 16, weight=-1.0)
    first(16, 18)
    first(16, 17, weight=-1.0)

    # A covered target is visible as colour 11 surrounded
    # by real/reflected piece filler.
    first(18, 11, weight=4.0)
    for dy, dx in directions:
        first(18, 4, dy, dx)
        first(18, 5, dy, dx)
    b1[18] = -7.0

    # Screen-edge exclusions.
    first(19, 16, weight=126.0)
    b1[19] = -124.0
    first(20, 17, weight=126.0)
    b1[20] = -124.0

    # Exact target signatures. At cell centres, distance-three
    # samples remain valid even when neighbouring targets are covered.
    for signature, required in enumerate(signatures):
        out = 6 + signature
        second(out, 10, weight=5.0)
        for direction, (dy, dx) in enumerate(directions):
            weight = 1.0 if direction in required else -1.0
            second(out, 6 + direction, dy, dx, weight=weight)
        second(out, 19, weight=-10.0)
        second(out, 20, weight=-10.0)
        b2[out] = -(5.0 + len(required) - 1.0)

    second(12, 11)
    second(13, 12)

    def target_mask(out):
        # Restore the full display cell around a covered marker.
        second(out, 10)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                second(out, 18, dy, dx)
        second(out, 18, weight=-1.0)
        second(out, 19, weight=-1.0)
        second(out, 20, weight=-1.0)

    # Target-gated coordinates for bounding-box extrema.
    for out, coordinate in (
        (14, 13), (15, 14), (16, 15), (17, 16)
    ):
        target_mask(out)
        second(out, coordinate)
        b2[out] = -1.0

    # Selected-axis coordinates.
    second(18, 11)
    second(18, 14)
    b2[18] = -1.0

    second(19, 12)
    second(19, 13)
    b2[19] = -1.0

    second(20, 18)

    # f3[0:4]: grouped piece signatures.
    # f3[4:8]: grouped target signatures.
    for group, members in enumerate(groups):
        for signature in members:
            p["W3"][group, signature] = 1.0
            p["W3"][4 + group, 6 + signature] = 1.0

            # Broad axial attraction for distant configurations.
            p["W3"][16 + group, 32 + 6 + signature] = 1.0
            p["W3"][20 + group, 64 + 6 + signature] = 1.0

    for out, channel in (
        (8, 12), (9, 13),
        (10, 14), (11, 15), (12, 16), (13, 17),
        (14, 18), (15, 19), (24, 20),
    ):
        p["W3"][out, channel] = 1.0

    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    # Period-three row and column lattices.
    # Zero padding fixes the phase; channels 1 and 3 identify
    # the centre pixels of the canonical 3x3 display cells.
    Wh[0, 0, 0, 1] = -1.0
    Wh[0, 1, 0, 1] = -1.0
    bh[0] = 1.0
    Wh[1, 0, 0, 1] = 1.0

    Wh[2, 2, 1, 0] = -1.0
    Wh[2, 3, 1, 0] = -1.0
    bh[2] = 1.0
    Wh[3, 2, 1, 0] = 1.0

    # Three channels per signature group:
    # source, diffusion field, and piece-gated field sample.
    for group in range(4):
        source = 4 + 3 * group
        field = source + 1
        match = source + 2

        Wh[source, 1, 1, 1] = 1.0
        Wh[source, 3, 1, 1] = 1.0
        Uh[source, 4 + group] = 1.0
        Uh[source, 24] = -0.1
        bh[source] = -2.0

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy or dx:
                    Wh[field, field, dy + 1, dx + 1] = 0.12375
        Wh[field, source, 1, 1] = 1.0
        Uh[field, 16 + group] = 0.002
        Uh[field, 20 + group] = 0.002

        Wh[match, field, 1, 1] = 1.0
        Wh[match, 1, 1, 1] = 8.0
        Wh[match, 3, 1, 1] = 8.0
        Uh[match, group] = 8.0
        bh[match] = -24.0

    # g = [h_max, h_mean, f3_max, f3_mean].
    Wz, bz = p["Wz"], p["bz"]
    for group, weight in enumerate((4.0, 4.0, 1.0, 1.0)):
        match = 6 + 3 * group
        Wz[0, 16 + match] = 4096.0 * 0.001 * weight

    # Explicit reward for covering goal-cell centres.
    Wz[0, 64 + 24] = 4096.0 * 0.002

    Wz[1, 32 + 8] = 1.0
    Wz[2, 32 + 9] = 1.0

    axis_gate = 4.0
    deadband = 0.05

    # Axis movement requests: up, down, left, right.
    Wz[3, 32 + 15] = 2.0
    Wz[3, 32 + 10] = -1.0
    Wz[3, 32 + 12] = 1.0
    Wz[3, 32 + 9] = axis_gate
    bz[3] = -1.0 - deadband - axis_gate

    Wz[4, 32 + 10] = 1.0
    Wz[4, 32 + 12] = -1.0
    Wz[4, 32 + 15] = -2.0
    Wz[4, 32 + 9] = axis_gate
    bz[4] = 1.0 - deadband - axis_gate

    Wz[5, 32 + 14] = 2.0
    Wz[5, 32 + 11] = -1.0
    Wz[5, 32 + 13] = 1.0
    Wz[5, 32 + 8] = axis_gate
    bz[5] = -1.0 - deadband - axis_gate

    Wz[6, 32 + 11] = 1.0
    Wz[6, 32 + 13] = -1.0
    Wz[6, 32 + 14] = -2.0
    Wz[6, 32 + 8] = axis_gate
    bz[6] = 1.0 - deadband - axis_gate

    Wm, Wmz, Wma, bm = (
        p["Wm"], p["Wmz"], p["Wma"], p["bm"]
    )
    sharpness = 10.0

    # Previous score, kept in tanh's nearly linear range.
    Wmz[0, 0] = 0.0001

    def score_test(slot, sign):
        Wmz[slot, 0] = sign * 1.0e9
        Wm[slot, 0] = -sign * 1.0e13
        bm[slot] = -8.0

    score_test(1, 1.0)
    score_test(24, -1.0)

    # Last movement direction, latched across undo.
    # Slots 6..9 indicate the immediately preceding movement.
    for direction in range(4):
        slot = 2 + direction
        Wm[slot, slot] = sharpness
        for previous in range(4):
            Wma[slot, 1 + previous] = (
                3.0 * sharpness
                if previous == direction
                else -3.0 * sharpness
            )

        Wma[6 + direction, 1 + direction] = 2.0 * sharpness
        bm[6 + direction] = -sharpness

    # Begin a sweep after selection or at episode start.
    bm[10] = sharpness
    for action in (1, 2, 3, 4, 7):
        Wma[10, action] = -2.0 * sharpness

    # Undo advances to the next direction.
    for slot, predecessor in ((11, 2), (12, 3), (13, 4), (14, 5)):
        Wm[slot, predecessor] = sharpness
        Wma[slot, 7] = 2.0 * sharpness
        bm[slot] = -2.0 * sharpness

    for action in (1, 2, 3, 4):
        Wma[15, action] = 2.0 * sharpness
    bm[15] = -sharpness

    # Direction-specific improvement and deterioration tests.
    # Inactive directions are strongly suppressed.
    for direction in range(4):
        for slot, sign in (
            (16 + direction, 1.0),
            (20 + direction, -1.0),
        ):
            score_test(slot, sign)
            bm[slot] -= 1.0e8
            Wma[slot, 1 + direction] = 1.0e8

    Wo, bo = p["Wo"], p["bo"]
    next_slots = [10, 11, 12, 13]

    for direction in range(4):
        action = 1 + direction

        # Continue an improving translation.
        Wo[action, 128 + 16 + direction] = 3.0

        # Start after selection or the preceding undo.
        Wo[action, 128 + next_slots[direction]] = 2.0

        if direction:
            predecessor = direction - 1
            # Advance directly after an ineffective translation.
            Wo[action, 128 + 16 + predecessor] -= 1.0
            Wo[action, 128 + 20 + predecessor] -= 1.0
            Wo[action, 128 + 6 + predecessor] += 2.0

        # Selected axes use bounding-box alignment instead.
        Wo[action, 1] = -20.0
        Wo[action, 2] = -20.0
        Wo[action, 3 + direction] = 1000.0

    # Undo actual deterioration, not an ineffective boundary move.
    Wo[7, 128 + 24] = 3.0
    Wo[7, 128 + 15] = 5.0
    bo[7] = -5.0
    Wo[7, 1] = -20.0
    Wo[7, 2] = -20.0

    # Cycle selection after the final direction finishes.
    Wo[5, 128 + 14] = 2.0
    Wo[5, 128 + 19] = -1.0
    Wo[5, 128 + 23] = -1.0
    Wo[5, 128 + 9] = 2.0
    bo[5] = -2.0

    # An aligned axis also yields selection.
    Wo[5, 1] = 8.0
    Wo[5, 2] = 8.0

    bo[0] = -1000.0
    bo[6] = -1000.0

    return p
