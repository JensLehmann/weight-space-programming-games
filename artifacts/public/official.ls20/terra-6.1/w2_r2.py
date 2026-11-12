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
    W3 = p["W3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]
    Wz, bz = p["Wz"], p["bz"]
    Wmz, Wma, bm = p["Wmz"], p["Wma"], p["bm"]
    Wo, bo = p["Wo"], p["bo"]

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    colors = (12, 9, 14, 8)

    def pix1(out, color, dy, dx, weight=1.0):
        W1[out, color, dy + 2, dx + 2] += weight

    def pix2(out, inp, dy=0, dx=0, weight=1.0):
        W2[out, inp, dy + 1, dx + 1] += weight

    # Player center.
    for r in range(5):
        color = 12 if r < 2 else 9
        for c in range(5):
            pix1(0, color, r - 2, c - 2, 2.0)
    b1[0] = -49.0

    # Recharge ring.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                pix1(1, 11, dy, dx, 2.0)
    b1[1] = -15.0

    # Walls and destination lookahead.
    pix1(4, 4, 0, 0)
    for k, (dy, dx) in enumerate(directions):
        pix1(20 + k, 4, 2 * dy, 2 * dx)

    # Goal shape code.
    for r in range(3):
        for c in range(3):
            bit = float(1 << (3 * r + c))
            for color in colors:
                pix1(5, color, r - 1, c - 1, bit)
    for dy in (-2, 2):
        for dx in (-2, 2):
            pix1(5, 5, dy, dx, 512.0)
    b1[5] = -2048.0

    # A clipped goal detector, implemented as the difference
    # of two ReLUs in the next layer.
    for out in (6, 7):
        for r in range(3):
            for c in range(3):
                for color in colors:
                    pix1(out, color, r - 1, c - 1)
        for dy in (-2, 2):
            for dx in (-2, 2):
                pix1(out, 5, dy, dx, 10.0)
    b1[6] = -44.0
    b1[7] = -45.0

    def template(out, pattern):
        for r in range(3):
            for c in range(3):
                pix1(out, pattern[r][c], r - 1, c - 1, 2.0)
        b1[out] = -17.0

    template(8, [[3, 0, 3], [1, 0, 0], [3, 1, 3]])
    template(9, [[3, 0, 3], [3, 0, 0], [3, 0, 3]])
    template(10, [[9, 14, 14], [9, 0, 8], [12, 12, 8]])

    # Shape code for the enlarged status icon.
    for r in range(3):
        for c in range(3):
            bit = float(1 << (3 * r + c))
            for color in colors:
                pix1(11, color, 2 * (r - 1), 2 * (c - 1), bit)

    # Select the fixed status-icon center.
    pix1(12, 17, 0, 0, 63.0)
    b1[12] = -5.1
    pix1(13, 17, 0, 0, -63.0)
    b1[13] = 4.9
    pix1(14, 16, 0, 0, 63.0)
    b1[14] = -57.1
    pix1(15, 16, 0, 0, -63.0)
    b1[15] = 56.9

    # Color codes.
    for r in range(3):
        for c in range(3):
            for label, color in enumerate(colors, 1):
                pix1(16, color, r - 1, c - 1, float(label))
                pix1(17, color, 2 * (r - 1), 2 * (c - 1),
                     float(label))
    for dy in (-2, 2):
        for dx in (-2, 2):
            pix1(16, 5, dy, dx, 64.0)
    b1[16] = -256.0

    # Exclude the lower-left status display from goal detection.
    pix1(18, 16, 0, 0, 63.0)
    b1[18] = -54.1
    pix1(19, 17, 0, 0, -63.0)
    b1[19] = 8.9

    # Direction markers, suppressed by a visible wall ahead.
    for k, (dy, dx) in enumerate(directions):
        pix2(k, 0, -dy, -dx)
        pix2(k, 20 + k, weight=-2.0)

    pix2(4, 4)

    # Goal seed.
    pix2(5, 6)
    pix2(5, 7, weight=-1.0)
    pix2(5, 18, weight=-100.0)
    pix2(5, 19, weight=-100.0)

    # Switch seed.
    pix2(6, 8)
    pix2(6, 9)
    pix2(6, 10)

    # Recharge seed.
    pix2(7, 1)
    pix2(7, 18, weight=-100.0)

    # Goal shape code, gated by the goal detector.
    pix2(8, 5)
    pix2(8, 6, weight=512.0)
    pix2(8, 7, weight=-512.0)
    pix2(8, 18, weight=-1024.0)
    pix2(8, 19, weight=-1024.0)
    b2[8] = -512.0

    # Status shape code.
    pix2(9, 11)
    for inp in (12, 13, 14, 15):
        pix2(9, inp, weight=-1024.0)

    # Goal color code.
    pix2(10, 16)
    pix2(10, 6, weight=64.0)
    pix2(10, 7, weight=-64.0)
    pix2(10, 18, weight=-1024.0)
    pix2(10, 19, weight=-1024.0)
    b2[10] = -64.0

    # Status color code.
    pix2(11, 17)
    for inp in (12, 13, 14, 15):
        pix2(11, inp, weight=-1024.0)

    for i in range(8):
        W3[i, i] = 1.0
    W3[8, 8] = 1.0 / 512.0
    W3[9, 9] = 1.0 / 512.0
    W3[10, 10] = 1.0 / 64.0
    W3[11, 11] = 1.0 / 64.0

    # Bounded directional fields for goals and switches.
    # Straight propagation is stronger than turning.
    for base, seed in ((0, 5), (4, 6)):
        for d, (dy, dx) in enumerate(directions):
            out = base + d
            for q in range(4):
                weight = 0.94 if q == d else 0.018
                Wh[out, base + q, dy + 1, dx + 1] = weight
            Uh[out, seed] = 0.04
            Uh[out, 7] = 0.004
            Uh[out, 4] = -10.0

    # Directional samples near the player. The relatively small
    # gate preserves more precision than the preceding revision.
    gate = 0.5
    for k, (dy, dx) in enumerate(directions):
        for out, base in ((8 + k, 0), (12 + k, 4)):
            for q in range(4):
                Wh[out, base + q, dy + 1, dx + 1] = 1.0
            Uh[out, k] = gate
            bh[out] = -gate

    # Copy directional field scores and legal-looking markers to z.
    for k in range(4):
        Wz[k, 8 + k] = 1.0
        Wz[4 + k, 12 + k] = 1.0
        Wz[8 + k, 32 + k] = 1.0

    # f3 maxima are at g[32:64].
    goal_shape, status_shape = 40, 41
    goal_color, status_color = 42, 43

    # For each difference, ReLU(d)-ReLU(d-1), together with
    # the analogous negative part, gives a clipped mismatch signal.
    def mismatch_units(start, left, right, scale):
        Wz[start, left] = scale
        Wz[start, right] = -scale

        Wz[start + 1, left] = scale
        Wz[start + 1, right] = -scale
        bz[start + 1] = -1.0

        Wz[start + 2, left] = -scale
        Wz[start + 2, right] = scale

        Wz[start + 3, left] = -scale
        Wz[start + 3, right] = scale
        bz[start + 3] = -1.0

    mismatch_units(12, goal_shape, status_shape, 512.0)
    mismatch_units(16, goal_color, status_color, 64.0)

    def add_mismatch(memory_index, start, coefficient):
        Wmz[memory_index, start] += coefficient
        Wmz[memory_index, start + 1] -= coefficient
        Wmz[memory_index, start + 2] += coefficient
        Wmz[memory_index, start + 3] -= coefficient

    # Memory units 0..3: goal scores, suppressed on either mismatch.
    # Units 4..7 and 8..11: switch scores, activated by a shape
    # mismatch or a color mismatch respectively.
    for k in range(4):
        Wmz[k, k] = 3.0
        add_mismatch(k, 12, -20.0)
        add_mismatch(k, 16, -20.0)

        Wmz[4 + k, 4 + k] = 3.0
        bm[4 + k] = -20.0
        add_mismatch(4 + k, 12, 20.0)

        Wmz[8 + k, 4 + k] = 3.0
        bm[8 + k] = -20.0
        add_mismatch(8 + k, 16, 20.0)

        action = k + 1
        Wo[action, 128 + k] = 1.0
        Wo[action, 128 + 4 + k] = 1.0
        Wo[action, 128 + 8 + k] = 1.0

        # Favor directions whose marker survived wall detection.
        Wo[action, 8 + k] = 0.01

    # Small penalty for immediately reversing the preceding action.
    for k in range(4):
        Wma[12 + k, k + 1] = 1.0
    opposite_actions = (2, 1, 4, 3)
    for k, action in enumerate(opposite_actions):
        Wo[action, 128 + 12 + k] = -2e-6

    for action in (0, 5, 6, 7):
        bo[action] = -1000.0

    return p
