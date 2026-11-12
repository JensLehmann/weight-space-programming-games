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

    W1, b1 = p["W1"], p["b1"]
    W2, b2 = p["W2"], p["b2"]
    W3, b3 = p["W3"], p["b3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]
    Wz, bz = p["Wz"], p["bz"]
    Wmz, bm, Wo = p["Wmz"], p["bm"], p["Wo"]

    colours = (12, 9, 14, 8)
    offsets = [
        (r, c) for r in (-1, 0, 1)
        for c in (-1, 0, 1)
    ]
    outer = [i for i, rc in enumerate(offsets) if rc != (0, 0)]
    corners = [
        k for k, i in enumerate(outer)
        if abs(offsets[i][0]) == 1
        and abs(offsets[i][1]) == 1
    ]
    ring = [
        (r, c) for r in range(-2, 3)
        for c in range(-2, 3)
        if abs(r) == 2 or abs(c) == 2
    ]

    def c1(out, inp, dr=0, dc=0, value=1.0):
        W1[out, inp, dr + 2, dc + 2] += value

    def c2(out, inp, value=1.0, dr=0, dc=0):
        W2[out, inp, dr + 1, dc + 1] += value

    def exact_patch(out, image):
        height, width = len(image), len(image[0])
        for r in range(height):
            for c in range(width):
                c1(
                    out, image[r][c],
                    r - height // 2, c - width // 2,
                )
        b1[out] = 1 - height * width

    # Goal peripheral pixels, conditional on the surrounding black ring.
    for out, (dr, dc) in enumerate(offsets):
        if out == 4:
            continue
        for colour in colours:
            c1(out, colour, dr, dc)
        for rr, cc in ring:
            c1(out, 5, rr, cc)
        b1[out] = -16.0

    # Independent goal-centre detector.
    for dr, dc in ring:
        c1(4, 5, dr, dc)
    b1[4] = -15.0

    # Peripheral pixels of the doubled status icon.
    for k, i in enumerate(outer):
        dr, dc = offsets[i]
        for colour in colours:
            c1(9 + k, colour, 2 * dr, 2 * dc)

    # Two-bit colour measurements for current and target icons.
    for bit in range(2):
        for ci, colour in enumerate(colours):
            if (ci >> bit) & 1:
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        c1(17 + bit, colour, dr, dc)
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        c1(19 + bit, colour, dr, dc)
        for dr, dc in ring:
            c1(19 + bit, 5, dr, dc, 10.0)
        b1[19 + bit] = -160.0

    # Budget-bar pixels, excluding refill icons higher on the board.
    c1(21, 11)
    c1(21, 16)
    b1[21] = -1.95

    # Positional hinges isolate the status-icon sampling position.
    c1(23, 16, value=-63.0)
    b1[23] = 57.0
    c1(24, 17, value=63.0)
    b1[24] = -5.0
    c1(25, 16, value=63.0)
    c1(25, 17, value=-63.0)
    b1[25] = -47.0

    exact_patch(26, [
        [12, 12, 12, 12, 12],
        [12, 12, 12, 12, 12],
        [9, 9, 9, 9, 9],
        [9, 9, 9, 9, 9],
        [9, 9, 9, 9, 9],
    ])
    exact_patch(27, [
        [3, 3, 3, 3, 3],
        [3, 0, 3, 3, 3],
        [3, 3, 0, 0, 3],
        [3, 3, 0, 3, 3],
        [3, 3, 3, 3, 3],
    ])
    exact_patch(28, [
        [3, 3, 3, 3, 3],
        [3, 9, 14, 14, 3],
        [3, 9, 0, 8, 3],
        [3, 12, 12, 8, 3],
        [3, 3, 3, 3, 3],
    ])
    exact_patch(29, [
        [3, 3, 3, 3, 3],
        [3, 3, 0, 3, 3],
        [3, 1, 0, 0, 3],
        [3, 3, 1, 3, 3],
        [3, 3, 3, 3, 3],
    ])
    exact_patch(30, [
        [11, 11, 11],
        [11, 3, 11],
        [11, 11, 11],
    ])

    # Dilate walls by the player half-width.
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            c1(31, 4, dr, dc, 100.0)
    c1(31, 16)

    def hud_gate(out, bias):
        c2(out, 23, -200.0)
        c2(out, 24, -200.0)
        c2(out, 25, -100.0)
        b2[out] = bias

    for k, i in enumerate(outer):
        c2(k, i)
        c2(k, 25, -100.0)
        c2(8 + k, 9 + k)
        hud_gate(8 + k, 500.0)

    for bit in range(2):
        c2(16 + bit, 19 + bit)
        c2(16 + bit, 25, -100.0)
        c2(18 + bit, 17 + bit)
        hud_gate(18 + bit, 500.0)
        c2(20 + bit, 17 + bit)
        hud_gate(20 + bit, 499.0)

    for out, inp in [
        (22, 26), (23, 27), (24, 28),
        (25, 29), (26, 30), (27, 31),
    ]:
        c2(out, inp)
    b2[27] = -2.0

    # Priority carrier: larger at upper goals.
    c2(28, 4, 64.0)
    c2(28, 31, -64.0)
    c2(28, 25, -10000.0)

    # Directional payloads at the player's four immediate neighbours.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    codes = (0.125, 0.375, 0.625, 0.875)
    for (dr, dc), code in zip(directions, codes):
        c2(29, 26, 1.0, -dr, -dc)
        c2(30, 26, code, -dr, -dc)
    c2(31, 21)

    # Bind pooled attributes to a common target priority.
    for k in range(8):
        W3[k, 28] = 1.0
        W3[k, k] = 0.25
        W3[8 + k, 8 + k] = 1.0

    for bit in range(2):
        W3[16 + bit, 28] = 1.0
        W3[16 + bit, 16 + bit] = 0.025
        W3[18 + bit, 18 + bit] = 1.0
        W3[18 + bit, 20 + bit] = -1.0

    # Only the uppermost goal in a column seeds the goal wave.
    W3[20, 28] = 1.0
    W3[20, 64 + 28] = -1.0
    W3[20, :8] = 0.25

    for channel in range(22, 32):
        W3[channel, channel] = 1.0

    # Exclude blocked neighbours from pooled directional readouts.
    W3[29, 27] = -100.0
    W3[30, 27] = -100.0

    # Four capped reachability waves and paired arrival-time counters.
    # Modes: goal/refill, shape, colour, rotation.
    source_channels = (20, 23, 24, 25)
    for mode, source in enumerate(source_channels):
        a, b, d, e = [4 * mode + k for k in range(4)]
        for out in (a, b):
            for dr, dc in [
                (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)
            ]:
                Wh[out, a, dr + 1, dc + 1] = 1.0
                Wh[out, b, dr + 1, dc + 1] = -1.0
            Uh[out, source] = 1.0
            if mode == 0:
                Uh[out, 26] = 1.0
            Uh[out, 27] = -8.0
        bh[b] = -1.0

        for out in (d, e):
            Wh[out, out, 1, 1] = 1.0
            Wh[out, a, 1, 1] = 1.0
            Wh[out, b, 1, 1] = -1.0
            Uh[out, 29] = 2.0
        Uh[e, 30] = 1.0 / 128.0

    def fg(channel):
        return 32 + channel

    def goal_pixel(unit, pixel, weight):
        Wz[unit, fg(pixel)] += 4.0 * weight
        Wz[unit, fg(28)] -= 4.0 * weight

    # Absolute deviation from each possible directional payload.
    for mode in range(4):
        d, e = 4 * mode + 2, 4 * mode + 3
        for action, code in enumerate(codes):
            i = 4 * mode + action
            Wz[2 * i, e] = 1.0
            Wz[2 * i, d] = -1.0
            bz[2 * i] = -code
            Wz[2 * i + 1, e] = -1.0
            Wz[2 * i + 1, d] = 1.0
            bz[2 * i + 1] = code

    # Shape invariants distinguish all six shapes regardless of rotation.
    shape_units = []
    unit = 32
    for indices in (range(8), corners):
        for sign in (1.0, -1.0):
            for k in indices:
                goal_pixel(unit, k, sign)
                Wz[unit, fg(8 + k)] -= sign
            bz[unit] = -0.1
            shape_units.append(unit)
            unit += 1

    colour_units = []
    for bit in range(2):
        goal_channel = 16 + bit
        current_channel = 18 + bit
        u = 36 + 2 * bit

        Wz[u, fg(goal_channel)] = 5.0
        Wz[u, fg(28)] = -5.0
        Wz[u, fg(current_channel)] = -1.0
        bz[u] = -0.1

        Wz[u + 1, fg(current_channel)] = 1.0
        Wz[u + 1, fg(goal_channel)] = -20.0
        Wz[u + 1, fg(28)] = 20.0
        bz[u + 1] = -0.1
        colour_units.extend((u, u + 1))

    rotation_units = []
    unit = 40
    for k in range(8):
        for sign in (1.0, -1.0):
            goal_pixel(unit, k, sign)
            Wz[unit, fg(8 + k)] -= sign
            bz[unit] = -0.1
            rotation_units.append(unit)
            unit += 1

    # Current remaining budget, measured from the rendered bar.
    bar_per_step = (
        (61.0 / 63.0 - 0.95)
        + (62.0 / 63.0 - 0.95)
    )
    budget_scale = 4096.0 / bar_per_step
    Wz[56, 64 + 31] = budget_scale

    # A clipped low-budget indicator, conditional on a visible refill.
    # Their difference is 0.25 when the budget is at most 12.
    for u, threshold in [(57, 12.5), (58, 12.25)]:
        Wz[u, 64 + 31] = -budget_scale
        Wz[u, fg(26)] = 100.0
        bz[u] = threshold - 100.0

    gate = 1000.0
    mem_unit = 0

    def navigation_unit(mode, action, suppress, emergency=False):
        nonlocal mem_unit
        u = mem_unit
        mem_unit += 1
        i = 4 * mode + action
        Wmz[u, 2 * i] = -24.0
        Wmz[u, 2 * i + 1] = -24.0
        bm[u] = 3.0
        for cost_unit in suppress:
            Wmz[u, cost_unit] -= gate

        if emergency:
            # Enable only when the clipped emergency indicator is one.
            Wmz[u, 57] += 4.0 * gate
            Wmz[u, 58] -= 4.0 * gate
            bm[u] -= gate
        else:
            Wmz[u, 57] -= 4.0 * gate
            Wmz[u, 58] += 4.0 * gate
        return u

    # Telescoping gates implement current-observation priority without
    # requiring delayed binary flags in recurrent memory.
    #
    # For example, shape navigation minus shape navigation suppressed
    # by shape mismatch is zero when matched and active when mismatched.
    sc = shape_units + colour_units
    scr = sc + rotation_units

    for action in range(4):
        goal = navigation_unit(0, action, scr)
        Wo[1 + action, 128 + goal] += 1.0

        shape_a = navigation_unit(1, action, [])
        shape_b = navigation_unit(1, action, shape_units)
        Wo[1 + action, 128 + shape_a] += 1.0
        Wo[1 + action, 128 + shape_b] -= 1.0

        colour_a = navigation_unit(2, action, shape_units)
        colour_b = navigation_unit(2, action, sc)
        Wo[1 + action, 128 + colour_a] += 1.0
        Wo[1 + action, 128 + colour_b] -= 1.0

        rotation_a = navigation_unit(3, action, sc)
        rotation_b = navigation_unit(3, action, scr)
        Wo[1 + action, 128 + rotation_a] += 1.0
        Wo[1 + action, 128 + rotation_b] -= 1.0

        refill = navigation_unit(0, action, [], emergency=True)
        Wo[1 + action, 128 + refill] += 1.0

    p["bo"][:] = -1000.0
    p["bo"][1:5] = 0.0
    return p
