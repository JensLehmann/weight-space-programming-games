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

    colors = (10, 6, 15, 11, 12, 8)
    widths = (1, 2, 3, 4, 5, 7)

    def k1(out, inp, dy, dx, value):
        W1[out, inp, dy + 2, dx + 2] += value

    def k2(out, inp, dy, dx, value):
        W2[out, inp, dy + 1, dx + 1] += value

    def center2(out, inp, value=1.0):
        k2(out, inp, 0, 0, value)

    def square2(out, inp, value=1.0):
        W2[out, inp] += value

    def play_only(out):
        center2(out, 19, -1000.0)

    def header_only(out):
        center2(out, 20, -1000.0)
        center2(out, 21, -1000.0)

    def gaussian(n, sigma):
        t = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
        a = np.exp(-0.5 * (t / sigma) ** 2)
        a /= a.sum()
        return np.outer(a, a).astype(np.float32)

    g5 = gaussian(5, 1.15)
    g3 = gaussian(3, 0.80)

    # First convolution: symbols, sources, local geometry.

    # One anchor per solid header symbol.
    for rank, color in enumerate(colors):
        if rank == 0:
            k1(rank, color, 0, 0, 1.0)
        else:
            for dy, dx in ((-1, -1), (-1, 0), (0, -1), (0, 0)):
                k1(rank, color, dy, dx, 1.0)
            for dy, dx in ((-2, -1), (-2, 0), (-1, -2), (0, -2)):
                k1(rank, color, dy, dx, -4.0)
            b1[rank] = -3.0

    k1(6, 2, 0, 0, 1.0)

    skeleton = (
        (-2, 0),
        (-1, -1), (-1, 1),
        (0, -2), (0, 2),
        (1, -1), (1, 1),
    )

    def creature_detector(out, palette):
        for color in palette:
            for dy, dx in skeleton:
                k1(out, color, dy, dx, 1.0)
            k1(out, color, -2, -1, -4.0)
            k1(out, color, -2, 1, -4.0)
        b1[out] = -6.0

    creature_detector(7, (7, 14, 13))
    creature_detector(24, (14,))
    creature_detector(25, (13,))

    # Dense goal interiors, including grey tutorial markings.
    W1[8, 9] = 1.0
    W1[8, 3] = 1.0
    b1[8] = -24.0

    for rank, (color, width) in enumerate(zip(colors, widths)):
        group = rank % 3
        mass = 1.0 / float(width * width)

        k1(9 + group, color, 0, 0, mass)
        W1[12 + group, color] += 1.0
        W1[15 + group, color] += mass * g5

        if rank >= 3:
            W1[18, color] += 1.0

        # Exact rank at visible fruit pixels, without spatial summation.
        k1(23, color, 0, 0, float(rank))

        W1[29, color] = 1.0
        W1[30, color] = 1.0

    b1[30] = -1.0

    # Coordinate gates.
    k1(19, 16, 0, 0, -63.0)
    b1[19] = 10.0
    k1(20, 16, 0, 0, 63.0)
    b1[20] = -9.0
    k1(21, 17, 0, 0, -63.0)
    b1[21] = 26.0

    # Fruit surrounded by goal colour is an attenuated field source.
    for color in colors:
        k1(22, color, 0, 0, 1.0)
    W1[22, 9] += 0.05
    b1[22] = -1.4

    for color in (7, 14, 13):
        W1[26, color] = 1.0
        W1[31, color] = 1.0
    b1[31] = -1.0
    W1[27, 14] = 1.0
    W1[28, 13] = 1.0

    # Second convolution: play-area masks and header measurements.
    for group in range(3):
        center2(group, 9 + group)
        square2(group, 22, -8.0)
        play_only(group)

        # Their difference is a clipped dilation.
        for out, bias in ((3 + group, 0.0), (6 + group, -1.0)):
            square2(out, 12 + group)
            b2[out] = bias
            play_only(out)

        W2[9 + group, 15 + group] = g3
        play_only(9 + group)

    for rank in range(6):
        center2(12, rank, float(rank + 1))
    header_only(12)

    for out, bias in ((13, 0.0), (14, -1.0)):
        square2(out, 18)
        b2[out] = bias
        play_only(out)

    center2(15, 8)
    play_only(15)

    center2(16, 7)
    play_only(16)

    for out, bias in ((17, 0.0), (18, -1.0)):
        square2(out, 26)
        b2[out] = bias
        play_only(out)

    center2(19, 6)
    header_only(19)

    for out, inp, bias in (
        (20, 27, 0.0),
        (21, 27, -1.0),
        (22, 28, 0.0),
        (23, 28, -1.0),
    ):
        square2(out, inp)
        b2[out] = bias
        play_only(out)

    center2(24, 7)
    center2(24, 24)
    center2(24, 25, 2.0)
    header_only(24)

    # Exact AND of two clipped local-presence measurements.
    center2(25, 29)
    center2(25, 30, -1.0)
    center2(25, 26)
    center2(25, 31, -1.0)
    b2[25] = -1.0
    play_only(25)

    center2(26, 19)
    center2(27, 20)
    b2[27] = -53.0

    center2(28, 23)
    play_only(28)

    # Axial mixing.
    def mask3(out, plus, minus, scale=1.0, axial=0):
        W3[out, axial + plus] += scale
        W3[out, axial + minus] -= scale

    for group in range(3):
        W3[group, group] = 1.0
        W3[group, 32 + group] = 0.025
        W3[group, 64 + group] = 0.025

        mask3(3 + group, 3 + group, 6 + group)
        W3[6 + group, 9 + group] = 1.0

    mask3(9, 13, 14)
    W3[10, 32 + 12] = 1.0
    W3[11, 15] = 1.0

    W3[12, 16] = 0.10
    W3[12, 32 + 16] = 0.06
    W3[12, 64 + 16] = 0.06

    mask3(13, 17, 18)
    W3[14, 19] = 1.0

    for group in range(3):
        mask3(15, 3 + group, 6 + group, float(group))
        mask3(16, 3 + group, 6 + group)
        mask3(17, 3 + group, 6 + group)

    mask3(15, 13, 14, 3.0)
    b3[17] = -1.0

    mask3(18, 17, 18)
    mask3(18, 20, 21)
    mask3(18, 22, 23, 2.0)
    W3[19, 32 + 24] = 1.0

    # Safe waiting uses a maximum of exact ranks, not a sum of masks.
    W3[20, 32 + 28] = 1.0

    W3[21, 26] = 1.0
    W3[22, 27] = 1.0

    # Danger row/column union. It extends a local danger detection
    # to escape clicks on the far side of the creature.
    for out in (25, 26):
        W3[out, 32 + 25] = 1.0
        W3[out, 64 + 25] = 1.0
    b3[26] = -1.0

    # Recurrent channels:
    # 0..2  compatible-fruit fields
    # 3..5  merge scores
    # 6,7   integrated clipped goal wavefront
    # 8     unrequested-creature escape score
    # 9     requested fruit rank + 1
    # 10    oversized-fruit waiting score
    # 11    fruit delivery score
    # 12    creature field
    # 13    requested-creature delivery score
    # 14    low-fruit-source gate
    # 15    requested creature rank
    cross = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))

    def hc(out, inp, value):
        Wh[out, inp, 1, 1] += value

    for channel, source in ((0, 0), (1, 1), (2, 2), (12, 12)):
        for dy, dx in cross:
            Wh[channel, channel, dy + 1, dx + 1] = 0.199
        Uh[channel, source] = 1.0

    # a-b is a binary reached-set. The common a(previous) term
    # also integrates its local coverage, producing a goal potential.
    for out in (6, 7):
        hc(out, 6, 1.0)
        for dy, dx in cross:
            Wh[out, 6, dy + 1, dx + 1] += 1.0
            Wh[out, 7, dy + 1, dx + 1] -= 1.0
        Uh[out, 11] = 1.0
    bh[7] = -1.0

    # Downward prefix scans broadcast the header requests.
    Wh[9, 9, 0, 1] = 1.0
    Uh[9, 10] = 1.0

    Wh[15, 15, 0, 1] = 1.0
    Uh[15, 19] = 1.0

    K = 32.0
    M = 256.0
    goal_reference = 635.0

    for group in range(3):
        out = 3 + group

        for other in range(3):
            hc(out, other, 1.0 if other == group else -0.4)

        hc(out, 12, -0.8)
        hc(out, 9, K)
        hc(out, 6, 0.0001)

        Uh[out, 15] = -K
        Uh[out, 6 + group] = -32.0
        Uh[out, 3 + group] = M
        Uh[out, 17] = -2.0 * M

        bh[out] = -1.5 * K - M - goal_reference * 0.0001

    # Escape is only enabled for unrequested creatures.
    # Negative fruit fields choose the side away from the cluster.
    escape_gate = 512.0
    Uh[8, 13] = escape_gate
    Uh[8, 25] = escape_gate
    Uh[8, 26] = -escape_gate
    Uh[8, 21] = -10000.0
    Uh[8, 22] = -10000.0
    hc(8, 15, -1024.0)
    for channel in range(3):
        hc(8, channel, -4.0)
    hc(8, 6, 0.0001)
    bh[8] = -2.0 * escape_gate + 256.0 - goal_reference * 0.0001

    # Oversized fruit takes priority over every movement branch.
    # Within its row, weak repulsion selects a distant waiting click.
    Uh[10, 20] = 1000.0
    hc(10, 9, -1000.0)
    Uh[10, 21] = -10000.0
    Uh[10, 22] = -10000.0
    bh[10] = 500.0
    for channel in (0, 1, 2, 12):
        hc(10, channel, -0.25)

    # Saturated source gate suppresses parked fruit.
    bh[14] = 0.18
    for channel in range(3):
        hc(14, channel, -1.0)

    # Delivery is restricted to the local fruit support.
    Uh[11, 16] = M
    Uh[11, 17] = -2.0 * M
    Uh[11, 15] = K
    hc(11, 9, -K)
    hc(11, 14, -80.0)
    hc(11, 6, 0.008)
    bh[11] = -M + K + 5.0 - goal_reference * 0.008

    Uh[13, 13] = 100.0
    Uh[13, 18] = -8.0
    hc(13, 15, 8.0)
    hc(13, 6, 0.0016)
    hc(13, 12, 0.03)
    bh[13] = 1.2 - goal_reference * 0.0016 - 100.0

    for channel in (3, 4, 5, 8, 10, 11, 13):
        p["Wc"][0, 32 + channel] = 1.0

    p["Wc"][0, 16] = 1.0e-5
    p["Wc"][0, 13] = 1.0e-6
    p["Wc"][0, 21] = -10000.0
    p["Wc"][0, 22] = -10000.0

    # Discrete action choice and short recovery memory.
    p["Wz"][0, 32 + 14] = 1.0

    p["bo"][:] = -100.0
    p["bo"][6] = 1.0
    p["bo"][7] = -2.0
    p["bo"][0] = -8.0

    p["Wo"][7, 0] = 10.0
    p["Wo"][0, 0] = 10.0

    p["Wm"][0, 0] = 0.99
    p["Wma"][0, 7] = 2.0
    p["Wo"][0, 128] = 10.0

    return p
