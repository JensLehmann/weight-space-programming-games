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

    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def c1(out, inp, value, dy=0, dx=0):
        p["W1"][out, inp, 2 + dy, 2 + dx] += value

    def c2(out, inp, value, dy=0, dx=0):
        p["W2"][out, inp, 1 + dy, 1 + dx] += value

    def recurrent(out, inp, value, dy=0, dx=0):
        p["Wh"][out, inp, 1 + dy, 1 + dx] += value

    def agent_colors(out, value):
        c1(out, 10, value)
        c1(out, 1, value)

    # Paired ramps clip the agent mask to the two display halves.
    for out, sign, intercept in ((0, -1, 63), (2, 1, -63)):
        for offset in range(2):
            agent_colors(out + offset, 128)
            c1(out + offset, 17, sign * 126)
            p["b1"][out + offset] = intercept - 128 - offset

    # Planning passability includes unselected movable blocks.
    for color in (5, 10, 1, 9):
        c1(4, color, 1)
    agent_colors(5, 1)

    # Agent-masked coordinates.
    agent_colors(6, 2)
    c1(6, 17, 1)
    p["b1"][6] = -2
    agent_colors(7, 2)
    c1(7, 16, 1)
    p["b1"][7] = -2

    # Dilation of block interiors; paired ReLUs cap this to one.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            c1(8, 9, 1, dy, dx)
            c1(9, 9, 1, dy, dx)
    p["b1"][9] = -1

    # Selected block patterns at common display cell scales.
    for out, lo, hi in ((10, -1, 2), (11, -2, 2)):
        c1(out, 11, 1)
        for dy, dx in ((lo, 0), (hi, 0), (0, lo), (0, hi)):
            c1(out, 5, 1, dy, dx)
        p["b1"][out] = -4

    switch_colors = (12, 14, 15)
    for color in switch_colors:
        c1(12, color, 1)

    # Four block-clearance samples.
    for out, (dy, dx) in (
            (21, (-2, 0)), (13, (2, 0)),
            (22, (0, -2)), (27, (0, 2))):
        c1(out, 5, 1, dy, dx)

    # Samples extend to distance three in the second convolution.
    for out, (dy, dx) in enumerate(
            ((0, -2), (0, 2), (-2, 0), (2, 0)), 15):
        for color in switch_colors:
            c1(out, color, 1, dy, dx)

    # Approximate narrow-patch entry mask, clipped by subtraction.
    for out, bias in ((19, -2), (20, -3)):
        for color in switch_colors:
            c1(out, color, 5)
            for dy, dx in ((0, -2), (0, 2), (-2, 0), (2, 0)):
                c1(out, color, -1, dy, dx)
        p["b1"][out] = bias

    # A triangular stripe at display column 32.
    for out, threshold in ((23, 31), (24, 32), (25, 33)):
        c1(out, 17, 63)
        p["b1"][out] = -threshold
    c1(26, 16, -1)
    p["b1"][26] = 1

    # Actual movement background excludes movable blocks.
    for color in (5, 10, 1):
        c1(28, color, 1)
    c1(29, 1, 1)

    # Exclude the upper UI row from rendezvous selection.
    c1(30, 16, 63)
    c1(31, 16, 63)
    p["b1"][30] = -0.5
    p["b1"][31] = -1.5

    # Eight legal leading-edge masks: four per display half.
    for side in range(2):
        ramp, excess = (0, 1) if side == 0 else (2, 3)
        for direction, (dy, dx) in enumerate(directions):
            out = 4 * side + direction
            c2(out, ramp, 1)
            c2(out, excess, -1)
            c2(out, 5, -1, dy, dx)
            c2(out, 28, 1, dy, dx)
            c2(out, 19, 1, dy, dx)
            c2(out, 20, -1, dy, dx)
            c2(out, 8, -1, dy, dx)
            c2(out, 9, 1, dy, dx)
            p["b2"][out] = -1

    c2(8, 4, 1)

    # Left x, right x, left y, right y.
    for out, coord, ramp, excess in (
            (9, 6, 0, 1), (10, 6, 2, 3),
            (11, 7, 0, 1), (12, 7, 2, 3)):
        c2(out, coord, 1)
        c2(out, ramp, 1)
        c2(out, excess, -1)
        p["b2"][out] = -1

    c2(13, 29, 1)
    c2(14, 8, 1)
    c2(14, 9, -1)
    c2(15, 10, 1)
    c2(15, 11, 1)

    for out, sample, dy, dx in (
            (16, 21, -1, 0), (17, 13, 1, 0),
            (18, 22, 0, -1), (19, 27, 0, 1)):
        c2(out, 10, 1)
        c2(out, 11, 1)
        c2(out, sample, 1, dy, dx)
        p["b2"][out] = -1

    # An isolated square must be narrow in BOTH axes.
    c2(20, 12, 1)
    c2(20, 15, -1, 0, -1)
    c2(20, 16, -1, 0, 1)
    c2(20, 17, -1, -1, 0)
    c2(20, 18, -1, 1, 0)
    c2(21, 12, 1)

    # Topmost traversable point on the central stripe.
    for out, multiplier in ((26, 4), (27, 1)):
        c2(out, 23, multiplier)
        c2(out, 24, -2 * multiplier)
        c2(out, 25, multiplier)
    c2(26, 4, 4)
    c2(26, 26, 1)
    c2(26, 30, 2)
    c2(26, 31, -2)
    p["b2"][26] = -10

    c2(28, 0, 1)
    c2(28, 1, -1)
    c2(29, 2, 1)
    c2(29, 3, -1)

    w3, b3 = p["W3"], p["b3"]
    for channel in range(8):
        w3[channel, channel] = 1

    # Walls remain blocked unless both a coloured patch and an
    # isolated-switch row are present. This opens switch interiors
    # without treating every narrow horizontal gate as traversable.
    w3[8, 8] = -1
    w3[8, 21] = -0.5
    w3[8, 32 + 20] = -0.5
    b3[8] = 1

    for channel in range(9, 20):
        w3[channel, channel] = 1
    w3[20, 20] = 1

    w3[23, 26] = 512
    w3[23, 64 + 26] = -512
    w3[23, 27] = 1
    b3[24] = 1
    w3[25, 28] = 1
    w3[26, 29] = 1

    # Two independent saturated wavefronts:
    # 0,1,2: rendezvous reach, excess, accumulated reach.
    # 3,4,5: switch reach, excess, accumulated reach.
    for base, source in ((0, 23), (3, 20)):
        reach, excess, arrival = base, base + 1, base + 2
        for out in (reach, excess):
            for dy, dx in directions:
                recurrent(out, reach, 2, dy, dx)
                recurrent(out, excess, -2, dy, dx)
            p["Uh"][out, source] = 2
            p["Uh"][out, 8] = -2048
        p["bh"][reach] = -0.5
        p["bh"][excess] = -1.5

        recurrent(arrival, arrival, 1)
        recurrent(arrival, reach, 1)
        recurrent(arrival, excess, -1)

    # A main-potential gradient plus a weaker switch gradient.
    # The baseline two exists only on legal agent boundary pixels.
    switch_weight = 0.15
    for side in range(2):
        for direction, (dy, dx) in enumerate(directions):
            mask = 4 * side + direction
            out = 6 + mask
            recurrent(out, 2, 1, dy, dx)
            recurrent(out, 2, -1)
            recurrent(out, 5, switch_weight, dy, dx)
            recurrent(out, 5, -switch_weight)
            p["Uh"][out, mask] = 512
            p["bh"][out] = -510

    # Arrival-time readouts at the agents.
    for out, mask in ((14, 25), (15, 26)):
        recurrent(out, 2, 1)
        p["Uh"][out, mask] = 512
        p["bh"][out] = -512

    # Global boundary SUMS, not maxima. Removing the legal-edge
    # baseline leaves a signed directional potential change.
    groups = ((0, 4), (1, 5), (3, 6), (2, 7))
    for z, masks in enumerate(groups):
        for mask in masks:
            p["Wz"][z, 16 + 6 + mask] = 1024
            p["Wz"][z, 64 + mask] = -2028
        p["bz"][z] = 4

    # Positive-gradient readouts for block-selection decisions.
    for z, masks in enumerate(groups, 8):
        for mask in masks:
            p["Wz"][z, 16 + 6 + mask] = 1024
            p["Wz"][z, 64 + mask] = -2048

    for z, feature in (
            (4, 9), (5, 10), (6, 13), (7, 14),
            (14, 11), (15, 12)):
        p["Wz"][z, 32 + feature] = 1

    for direction in range(4):
        p["Wz"][16 + direction, 32 + 16 + direction] = 1

    wm, wmz = p["Wm"], p["Wmz"]
    wma, bm = p["Wma"], p["bm"]

    # One initial probe, followed by an episode-long clock latch.
    wm[0, 0] = 5
    bm[0] = 2

    # Small coordinate memories are effectively linear.
    for unit, z in ((2, 4), (3, 5), (32, 14), (33, 15)):
        wmz[unit, z] = 0.001

    # Observe horizontal movement, gated by the previous action.
    cases = ((4, 1), (4, -1), (3, 1), (3, -1))
    for side in range(2):
        for offset, (action, sign) in enumerate(cases):
            unit = 4 + 4 * side + offset
            wmz[unit, 4 + side] = sign * 50
            wm[unit, 2 + side] = -sign * 50000
            wma[unit, action] = 25
            bm[unit] = -25

    wm[1, 1] = 5
    for offset, coefficient in enumerate((1, -1, -1, 1)):
        wm[1, 4 + offset] = 0.6 * coefficient
        wm[1, 8 + offset] = -0.6 * coefficient

    # In each pair, the irrelevant score saturates to one; the
    # relevant score remains near the linear part of tanh.
    # Positive orientation means ACTION4 is horizontally inward.
    for unit, score, orientation_sign in (
            (12, 2, -1), (13, 3, 1),
            (14, 3, -1), (15, 2, 1)):
        wmz[unit, score] = 0.1
        wm[unit, 1] = 20 * orientation_sign
        bm[unit] = 19.6

    wm[30, 1] = 20
    wm[31, 1] = -20
    bm[30] = bm[31] = 10

    # Selected-block movement persistence.
    for direction, action in enumerate((1, 2, 3, 4)):
        wmz[16 + direction, 6] = 5
        wma[16 + direction, action] = 5
        bm[16 + direction] = -8

    # Block-mode timer resets rapidly outside block mode.
    wm[20, 20] = 1
    wmz[20, 6] = 3
    bm[20] = -2.7

    # Select a block only after a cooldown and a route stall.
    wmz[21, 7] = 8
    wmz[21, 6] = -12
    wmz[21, 8:12] = -6
    wm[21, 23] = 3
    bm[21] = -9

    # End a block-relocation phase after several moves.
    wm[22, 20] = 10
    wmz[22, 6] = 8
    bm[22] = -13

    wm[23, 23] = 0.9
    wma[23, 6] = -5
    bm[23] = 0.25

    # Deterministic small tie-breaking oscillator.
    wm[24, 24] = 1.1
    wm[24, 25] = -0.4
    wm[25, 24] = 0.4
    wm[25, 25] = 1.1
    bm[24] = 0.02

    for direction, action in enumerate((1, 2, 3, 4)):
        wma[26 + direction, action] = 2

    # No-displacement detectors for the immediately previous move.
    for direction, action in enumerate((1, 2, 3, 4)):
        base = 34 + 3 * direction
        if action in (1, 2):
            current = ((14, 1), (15, 1))
            previous = ((32, 1), (33, 1))
        else:
            current = ((4, -1), (5, 1))
            previous = ((2, -1), (3, 1))

        for offset, sign in ((0, 1), (1, -1)):
            unit = base + offset
            for z, coefficient in current:
                wmz[unit, z] = sign * 50 * coefficient
            for memory, coefficient in previous:
                wm[unit, memory] = -sign * 50000 * coefficient
            wma[unit, action] = -100
            bm[unit] = 98

        wma[base + 2, action] = -100
        bm[base + 2] = 100

    wo, bo = p["Wo"], p["bo"]
    bo[:] = -1e6
    for action in (1, 2, 3, 4):
        bo[action] = 0

    wo[1, 0] = 1
    wo[2, 1] = 1
    bo[1] = bo[2] = -4

    for action, units in ((4, (12, 13)), (3, (14, 15))):
        for unit in units:
            wo[action, 128 + unit] = 10
        wo[action, 128 + 30] = -5
        wo[action, 128 + 31] = -5
        bo[action] = -10

    # Force only the initial horizontal probe.
    wo[4, 128] = -100
    bo[4] += 100.005

    opposite = {1: 2, 2: 1, 3: 4, 4: 3}
    for direction, action in enumerate((1, 2, 3, 4)):
        wo[action, 16 + direction] = 12
        wo[action, 128 + 16 + direction] = 2
        wo[opposite[action], 128 + 16 + direction] = -2
        wo[action, 128 + 26 + direction] = 0.08

        base = 34 + 3 * direction
        wo[action, 128 + base] = 2
        wo[action, 128 + base + 1] = 2
        wo[action, 128 + base + 2] = -4

    # Slight upward preference when parking movable blocks.
    wo[1, 6] = 0.25
    wo[1, 128 + 24] = 0.025
    wo[2, 128 + 24] = -0.025
    wo[3, 128 + 25] = 0.025
    wo[4, 128 + 25] = -0.025

    wo[6, 128 + 21] = 50
    wo[6, 128 + 22] = 50
    bo[6] = 50

    # Select blocks near the rendezvous route. In block mode, the
    # passive agent colour dominates, so the return click deselects.
    p["Wc"][0, 14] = 10000
    p["Wc"][0, 13] = 20000
    p["Wc"][0, 32 + 2] = 0.01

    return p
