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

    ink = (8, 9, 11, 15)
    G = 256.0

    def template(channel, mask, colours):
        mask = np.asarray(mask, dtype=bool)
        for colour in colours:
            p["W1"][channel, colour] = np.where(mask, 1.0, -1.0)
        p["b1"][channel] = 1.0 - float(mask.sum())

    def crop5(mask, anchor):
        ax, ay = anchor
        out = np.zeros((5, 5), dtype=bool)
        height, width = mask.shape
        for ky in range(5):
            for kx in range(5):
                xx = ax + kx - 2
                yy = ay + ky - 2
                if 0 <= xx < width and 0 <= yy < height:
                    out[ky, kx] = mask[yy, xx]
        return out

    # Actor templates: four orientations at each of two scales.
    actor = np.ones((4, 4), dtype=bool)
    actor[3, 1:3] = False
    actor_features = []

    for scale in (1, 2):
        mask0 = np.repeat(
            np.repeat(actor, scale, axis=0), scale, axis=1
        )
        n = 4 * scale
        anchor0 = (1, 1) if scale == 1 else (2, 6)

        for rotation in range(4):
            mask = np.rot90(mask0, -rotation)
            ax, ay = anchor0
            for _ in range(rotation):
                ax, ay = n - 1 - ay, ax

            channel = len(actor_features)
            template(channel, crop5(mask, (ax, ay)), ink)
            actor_features.append((channel, ax, ay, scale))

    # Goal templates. Anchor y=3 leaves enough receptive-field reach
    # to test the first unobscured row below the six-pixel goal glyph.
    goal = np.zeros((6, 6), dtype=bool)
    goal[1:, 0] = True
    goal[1:, 5] = True
    goal[4, 2:4] = True
    goal[5, :] = True

    goal_channels = []
    for colour in ink:
        for rotation in range(4):
            channel = 8 + len(goal_channels)
            mask = np.rot90(goal, -rotation)
            template(channel, crop5(mask, (2, 3)), (colour,))
            goal_channels.append(channel)

    # Coordinates in pixel units.
    p["W1"][24, 16, 2, 2] = 63.0
    p["W1"][25, 17, 2, 2] = 63.0

    # Editable horizontal and vertical program-mark centres.
    for channel, horizontal in ((26, True), (27, False)):
        if horizontal:
            marks = ((0, -1), (0, 0), (0, 1))
            blanks = ((-1, 0), (1, 0), (0, -2), (0, 2))
        else:
            marks = ((-1, 0), (0, 0), (1, 0))
            blanks = ((0, -1), (0, 1), (-2, 0), (2, 0))

        for dy, dx in marks:
            for colour in (1, 5):
                p["W1"][channel, colour, 2 + dy, 2 + dx] = 1.0
        for dy, dx in blanks:
            p["W1"][channel, 0, 2 + dy, 2 + dx] = 1.0

        p["b1"][channel] = -6.0

    p["W1"][28, 5, 2, 2] = 1.0

    # Solid interior of the Run button.
    p["W1"][29, 9, :, :] = 1.0
    p["b1"][29] = -24.0

    # Shared boundary/contact feature. Away from the top row this
    # has values 0, 1, or 2, not necessarily a Boolean.
    p["W1"][30, 18, 2, 2] = 1.0
    p["W1"][30, 18, 1, 2] = -1.0
    p["W1"][30, 6, 2, 4] = 1.0
    p["W1"][30, 6, 4, 2] = 1.0

    # Checkpoint checkerboard.
    yy, xx = np.indices((4, 4))
    checkpoint = ((xx + yy) % 2) == 0
    template(31, crop5(checkpoint, (1, 1)), (11,))

    def w2(out, source, weight, dy=0, dx=0):
        p["W2"][out, source, 1 + dy, 1 + dx] += weight

    # Sparse actor coordinates and excess scale.
    for channel, ax, ay, scale in actor_features:
        w2(0, channel, G - ax)
        w2(1, channel, G - ay)
        w2(17, channel, scale - 1.0)

    w2(0, 25, 1.0)
    w2(1, 24, 1.0)
    p["b2"][0:2] = -G

    # Goal interior starts at glyph position + (1,1).
    # With anchor (2,3), subtract (1,2).
    for channel in goal_channels:
        w2(2, channel, G - 1.0)
        w2(3, channel, G - 2.0)

    w2(2, 25, 1.0)
    w2(3, 24, 1.0)
    p["b2"][2:4] = -G

    # Control locations, current bits, and inverse coordinates.
    for channel in (26, 27):
        w2(9, channel, 1.0)
        w2(10, channel, 1.0)
        w2(11, channel, G)
        w2(12, channel, G)

    w2(10, 28, 1.0)
    p["b2"][10] = -1.0

    w2(11, 25, -1.0)
    w2(12, 24, -1.0)
    p["b2"][11:13] = 64.0 - G

    w2(13, 24, 1.0)
    w2(14, 25, 1.0)

    # Isolate the top-row impulse from ordinary wall contacts.
    w2(15, 30, 1.0)
    w2(15, 24, -1.0)

    w2(16, 29, 100.0)
    w2(16, 24, 1.0)
    w2(16, 25, 1.0)
    p["b2"][16] = -182.0

    # A checkerboard immediately left of a wall is a staging point.
    # For an unobstructed checkpoint, the second contact sample is
    # inside the checkpoint, so only the right-wall sample contributes.
    w2(18, 31, G)
    w2(18, 30, 1.0, dx=1)
    p["b2"][18] = -G

    # Emit candidate coordinates one row after the reset event.
    # The asymmetric gate is deliberate: even contact=2 cannot pass
    # without a matching checkpoint. Coordinate +80 supplies bias 20
    # after conversion to four-pixel movement units.
    for out, coordinate in ((19, 25), (20, 24)):
        w2(out, 31, 4.0 * G, dy=-1)
        w2(out, 30, G, dy=-1, dx=1)
        w2(out, coordinate, 1.0, dy=-1)
        p["b2"][out] = 79.0 - 5.0 * G

    # Goal approach contact, clipped to one by a hinge difference.
    for out, extra in ((21, 0.0), (22, 1.0)):
        for channel in goal_channels:
            w2(out, channel, G)
        w2(out, 30, 1.0, dy=1)
        p["b2"][out] = -G - extra

    # Axial pooling broadcasts sparse measurements along their rows.
    for channel in range(4):
        p["W3"][channel, 32 + channel] = 1.0

    p["W3"][4, 32 + 19] = 1.0
    p["W3"][5, 32 + 20] = 1.0
    p["W3"][6, 32 + 18] = 1.0
    p["W3"][7, 15] = 1.0

    # One-based instruction index relative to the editable panel.
    p["W3"][8, 14] = 0.2
    p["W3"][8, 32 + 11] = 0.2
    p["W3"][8, 9] = G
    p["b3"][8] = 1.0 - 64.0 / 5.0 - G

    p["W3"][9, 9] = 1.0
    p["W3"][10, 10] = 1.0

    # Three hinges per bit row make exact triangular row indicators.
    for bit in range(6):
        cuts = (3 * bit - 1, 3 * bit, 3 * bit + 1)
        for k, cut in enumerate(cuts):
            out = 11 + 3 * bit + k
            p["W3"][out, 13] = 1.0
            p["W3"][out, 64 + 12] = 1.0
            p["W3"][out, 9] = G
            p["b3"][out] = -64.0 - cut - G

    p["W3"][29, 16] = 0.002
    p["W3"][30, 32 + 17] = 1.0
    p["W3"][31, 32 + 21] = 1.0
    p["W3"][31, 32 + 22] = -1.0

    def local(out, terms):
        for source, weight in terms.items():
            p["Wh"][out, source, 1, 1] += weight

    def bit_term(out, bits, weight):
        for bit in bits:
            base = 11 + 3 * bit
            p["Uh"][out, base] += weight
            p["Uh"][out, base + 1] -= 2.0 * weight
            p["Uh"][out, base + 2] += weight

    M = 128.0
    B = 20.0

    # Downward scans:
    # H0,H1 = biased displacement to the selected endpoint.
    # H2 = one-cell lateral approach flag.
    # H3 = initial shrinking count.
    for out in range(4):
        p["Wh"][out, out, 0, 1] = 1.0

    p["Uh"][0, 0] = -0.25
    p["Uh"][0, 2] = 0.25
    p["Uh"][0, 4] = 0.25
    p["Uh"][0, 6] = -G
    p["Uh"][0, 7] = B
    p["Uh"][0, 31] = 1.0

    p["Uh"][1, 1] = -0.25
    p["Uh"][1, 3] = 0.25
    p["Uh"][1, 5] = 0.25
    p["Uh"][1, 6] = -G
    p["Uh"][1, 7] = B

    p["Uh"][2, 31] = 1.0
    p["Uh"][2, 6] = -G
    p["Uh"][3, 30] = 1.0

    # L=max(-dx,0), D=max(dy,0).
    local(4, {0: -1.0})
    p["bh"][4] = B

    local(5, {1: 1.0})
    p["bh"][5] = -B

    # R=L+dx, gated to the bit rows requiring it.
    local(6, {4: 1.0, 0: 1.0})
    bit_term(6, (0, 2, 3, 4, 5), M)
    p["bh"][6] = -B - M

    # Lower endpoint contribution L+shrink.
    local(9, {4: 1.0, 3: 1.0})
    bit_term(9, (1, 2, 3, 4, 5), M)
    p["bh"][9] = -M

    # Corrections for upward and downward instructions.
    local(10, {1: -1.0})
    bit_term(10, (1, 2, 3, 4), M)
    p["bh"][10] = B - M

    local(11, {1: 1.0})
    bit_term(11, (2, 3, 4, 5), M)
    p["bh"][11] = -B - M

    # Isolate the initial shrink opcode's bit 3.
    local(12, {4: 2.0, 0: 1.0, 3: M + 1.0})
    bit_term(12, (3,), M)
    p["bh"][12] = -B - 2.0 * M

    # Clipped additions: a final left move on bit 0, or shrink bit 3.
    for out, extra in ((7, 0.0), (8, 1.0)):
        local(out, {2: 1.0, 12: 2.0})
        p["Uh"][out, 9] = M
        bit_term(out, (1, 2, 4, 5), -M)
        bit_term(out, (3,), -1.0)
        p["bh"][out] = -M - extra

    # Desired-one intervals (lower, upper], separately for each bit.
    # The synthesized order is shrink, horizontal, vertical, return.
    upper = {
        3: 1.0,
        4: 2.0,
        0: 1.0,
        5: 2.0,
        1: -1.0,
        10: -1.0,
        11: -1.0,
        7: 1.0,
        8: -1.0,
        12: -1.0,
    }
    lower = {
        6: 1.0,
        9: 1.0,
        12: -1.0,
    }

    # Interval membership margin:
    # min(upper-j+.5, j-lower-.5).
    local(13, upper)
    local(13, lower)
    p["Uh"][13, 8] = -2.0
    p["bh"][13] = 1.0

    # Incorrect zero bit.
    local(14, upper)
    local(14, {13: -1.0})
    p["Uh"][14, 8] = -1.0
    p["Uh"][14, 9] = M
    p["Uh"][14, 10] = -M
    p["bh"][14] = 0.5 - M

    # Incorrect one bit.
    local(15, {
        source: -weight for source, weight in upper.items()
    })
    local(15, {13: 1.0})
    p["Uh"][15, 8] = 1.0
    p["Uh"][15, 9] = M
    p["Uh"][15, 10] = M
    p["bh"][15] = -0.5 - 2.0 * M

    # Repair a mismatching bit; otherwise click Run.
    p["bo"][6] = 10.0
    p["Wc"][0, 29] = 1.0
    p["Wc"][0, 32 + 14] = 1.0
    p["Wc"][0, 32 + 15] = 1.0

    return p
