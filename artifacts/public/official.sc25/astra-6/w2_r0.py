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

    def w1(o, c, dy=0, dx=0, v=1.0):
        p["W1"][o, c, dy + 2, dx + 2] += v

    def w2(o, c, dy=0, dx=0, v=1.0):
        p["W2"][o, c, dy + 1, dx + 1] += v

    def wh(o, c, dy=0, dx=0, v=1.0):
        p["Wh"][o, c, dy + 1, dx + 1] += v

    # Player templates. Anchor is one pixel below/right of its top-left.
    # Channels 0:4 are small; 4:8 are large, ordered up/down/left/right.
    base = np.array([[10, 10], [9, 9]], dtype=np.int32)
    rotations = (180, 0, 90, 270)
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))

    for large in (False, True):
        for d, rotation in enumerate(rotations):
            a = np.rot90(base, k=(-rotation // 90) % 4)
            if large:
                a = a.repeat(2, 0).repeat(2, 1)
            o = d + (4 if large else 0)
            for iy in range(a.shape[0]):
                for ix in range(a.shape[1]):
                    w1(o, int(a[iy, ix]), iy - 1, ix - 1)
            p["b1"][o] = 1 - a.size

            # Reject two-colour fragments of the exit as small players.
            if not large:
                if d < 2:
                    outside = [(y, x) for y in (-2, 1) for x in (-1, 0)]
                else:
                    outside = [(y, x) for y in (-1, 0) for x in (-2, 1)]
                for dy, dx in outside:
                    w1(o, 9, dy, dx, -1)
                    w1(o, 10, dy, dx, -1)

    # Spell icons: a coloured 2x2 block immediately below a UI border.
    for o, colour in ((8, 15), (9, 11), (10, 6)):
        for dy in (0, 1):
            for dx in (0, 1):
                w1(o, colour, dy, dx)
        w1(o, 3, -1, 0)
        p["b1"][o] = -4

    # Exit corners: three 9s and one 10 in a 2x2 square.
    for q in range(4):
        o = 11 + q
        for iy in range(2):
            for ix in range(2):
                colour = 10 if iy * 2 + ix == q else 9
                w1(o, colour, iy - 1, ix - 1)
        p["b1"][o] = -3

    # Exact switch-ring anchor, shared by both switch colours.
    for iy in range(4):
        for ix in range(4):
            if iy in (0, 3) or ix in (0, 3):
                w1(15, 6, iy - 1, ix - 1)
            else:
                w1(15, 12, iy - 1, ix - 1)
                w1(15, 13, iy - 1, ix - 1)
    p["b1"][15] = -15

    # Switch edges provide navigation sources on accessible approaches.
    w1(16, 6, v=4)
    for dy, dx in directions:
        w1(16, 12, dy, dx)
        w1(16, 13, dy, dx)
    p["b1"][16] = -4

    # Binary clearance channels.
    small_blockers = (3, 4, 5, 12, 13)
    large_blockers = (3, 4, 5, 6, 12, 13)
    for o, offsets, colours in (
        (17, (-1, 0), small_blockers),
        (18, (-1, 0, 1, 2), large_blockers),
        (19, (-1, 0, 1, 2), (4, 6)),
    ):
        p["b1"][o] = 1
        for dy in offsets:
            for dx in offsets:
                for colour in colours:
                    w1(o, colour, dy, dx, -1)

    # Coordinate hinges for the fixed spell-entry interface.
    rows = (50, 55, 60)
    cols = (25, 30, 35)
    for i, r in enumerate(rows):
        w1(20 + i, 16, v=63)
        p["b1"][20 + i] = -r
    for i, c in enumerate(cols):
        w1(23 + i, 17, v=63)
        p["b1"][23 + i] = -c
    w1(26, 16, v=63)
    w1(27, 17, v=63)
    for colour in (6, 11, 14, 15):
        w1(28, colour)

    # Second-layer navigation probes and facing-dependent interactions.
    for d, (dy, dx) in enumerate(directions):
        for c in range(8):
            w2(d, c, -dy, -dx)

        # At the forward probe, inspect the prospective footprint.
        w2(4 + d, d, -dy, -dx)
        w2(4 + d, 4 + d, -dy, -dx)
        w2(4 + d, 19, dy, dx, -1)

        w2(8 + d, 4 + d, -dy, -dx)
        w2(8 + d, 18, dy, dx, -1)

    for c in range(4, 8):
        w2(12, c)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            w2(13, 16, dy, dx)
            for c in range(11, 15):
                w2(13, c, dy, dx)

    w2(14, 17)       # Small-footprint clearance.
    w2(15, 15)       # Switch anchors.
    w2(16, 8)        # Resize icon.
    w2(17, 9)        # Teleport icon.
    w2(18, 10)       # Fire icon.
    w2(19, 28)       # Selected spell-grid pixels.

    # Nine one-cell coordinate tents.
    for r in range(3):
        for c in range(3):
            o = 20 + 3 * r + c
            w2(o, 26)
            w2(o, 27)
            w2(o, 20 + r, v=-2)
            w2(o, 23 + c, v=-2)
            p["b2"][o] = 1 - rows[r] - cols[c]

    # f3: navigation, local interaction flags, and row-broadcast summaries.
    p["W3"][0, 13] = 1
    p["W3"][1, 14] = -1
    p["b3"][1] = 1
    for d in range(4):
        p["W3"][2 + d, d] = 1
        p["W3"][6 + d, 4 + d] = 1
        p["W3"][10 + d, 8 + d] = 1
    p["W3"][14, 12] = 1

    # A = large-player flag + .1 * switch count + narrow-facing signal.
    p["W3"][15, 32 + 12] = 1
    p["W3"][15, 32 + 15] = 0.1
    for d in range(4):
        p["W3"][15, 32 + 8 + d] = 2

    p["W3"][16, 32 + 16] = 1   # Resize availability.
    p["W3"][17, 32 + 17] = 1   # Teleport availability.
    p["W3"][18, 32 + 18] = 100
    p["W3"][18, 32 + 15] = 1
    for d in range(4):
        p["W3"][18, 32 + 4 + d] = 10
    p["W3"][19, 15] = 1

    # Is any spell-entry cell active?
    p["W3"][20, 19] = 1
    p["W3"][20, 20:29] = 1
    p["b3"][20] = -1

    # Missing and extraneous bits for resize, teleport, and fire.
    patterns = (
        {1, 3, 5, 7},
        {0, 1, 4},
        {1, 4, 7},
    )
    for k, pattern in enumerate(patterns):
        missing = 21 + 2 * k
        extra = missing + 1
        for cell in range(9):
            if cell in pattern:
                p["W3"][missing, 20 + cell] = 1
            else:
                p["W3"][extra, 20 + cell] = 1
        p["W3"][missing, 19] = -1
        p["W3"][extra, 19] = 1
        p["b3"][extra] = -1

    # Harmonic navigation field, with hard obstacle suppression.
    for dy, dx in directions:
        wh(0, 0, dy, dx, 0.245)
    p["Uh"][0, 0] = 0.003
    p["Uh"][0, 1] = -100

    # Sample the field at four player-relative probes.
    for d in range(4):
        wh(1 + d, 0)
        p["Uh"][1 + d, 2 + d] = 100
        p["bh"][1 + d] = -100

    # Two one-way scans per row-summary channel. After 64 iterations,
    # down + up - local_summary is the same global sum at every cell.
    for k, field in enumerate((15, 16, 17, 18)):
        down, up = 5 + 2 * k, 6 + 2 * k
        wh(down, down, -1, 0)
        wh(up, up, 1, 0)
        p["Uh"][down, field] = 1
        p["Uh"][up, field] = 1

    def global_term(out, summary, coefficient):
        field = 15 + summary
        down, up = 5 + 2 * summary, 6 + 2 * summary
        wh(out, down, v=coefficient)
        wh(out, up, v=coefficient)
        p["Uh"][out, field] -= coefficient

    # Two resize cases: shrinking for narrow/switch routes, and growing
    # for the large teleport destination once switches are gone.
    global_term(13, 0, 100)
    global_term(13, 1, 2000)
    global_term(13, 2, -200)
    p["bh"][13] = -1905

    global_term(14, 0, -100)
    global_term(14, 1, 2000)
    global_term(14, 2, -200)
    p["bh"][14] = -1795

    # Fire is preferred at a forward grate/switch while switches remain.
    global_term(15, 3, 1000)
    global_term(15, 1, -5000)
    global_term(15, 2, -5000)
    p["bh"][15] = -100500

    mask_strength = 32768.0
    for out in (13, 14):
        p["Uh"][out, 21:23] += mask_strength
        p["bh"][out] -= mask_strength
    p["Uh"][15, 25:27] += mask_strength
    p["bh"][15] -= mask_strength

    # Teleport is the default click pattern; active resize/fire heads
    # override it. All targets are mismatches, not fixed action scripts.
    p["Wc"][0, 23:25] = 1
    p["Wc"][0, 32 + 13:32 + 16] = 1

    # Action readout.
    for d in range(4):
        p["Wz"][d, 1 + d] = 1

    # Cast when facing a firing obstruction with a live switch.
    for d in range(4):
        p["Wz"][4, 32 + 6 + d] = 1
    p["Wz"][4, 32 + 18] = 0.01
    p["Wz"][4, 32 + 19] = 1
    p["bz"][4] = -2.5

    # Cast when a large player's forward footprint is obstructed.
    for d in range(4):
        p["Wz"][5, 32 + 10 + d] = 1
    p["Wz"][5, 32 + 16] = 1
    p["bz"][5] = -1.5

    p["Wz"][6, 32 + 20] = -1
    p["bz"][6] = 1

    # One-step post-cast cooldown: previous action was a click and the
    # spell grid is now empty. Tanh is saturated to float32 +/-1.
    p["Wmz"][0, 6] = 20
    p["Wma"][0, 6] = 20
    p["bm"][0] = -30

    p["bo"][:] = -100
    for d in range(4):
        p["Wo"][1 + d, d] = 1
        p["bo"][1 + d] = -1e-35

    p["Wo"][6, 4] = 20
    p["Wo"][6, 5] = 20
    p["Wo"][6, 128] = -25
    p["bo"][6] = -25

    return p
