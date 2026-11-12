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

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    direction_index = {d: i for i, d in enumerate(directions)}

    def w1(out, inp, dy=0, dx=0, value=1.0):
        p["W1"][out, inp, dy + 2, dx + 2] += value

    def w2(out, inp, dy=0, dx=0, value=1.0):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    def wh(out, inp, dy=0, dx=0, value=1.0):
        p["Wh"][out, inp, dy + 1, dx + 1] += value

    # ---------------------------------------------------------------
    # First convolution: local appearance predicates.
    # ---------------------------------------------------------------
    FLOOR = 0
    PLAYER = 1
    SHIFTED_PLAYER = 2
    EXIT = 6
    DOOR = 7
    PURPLE_DOOR = 8
    RECORD_SWITCH = 9
    PURPLE_SWITCH = 10
    RECORD_WIRE = 11
    WHITE_SAMPLE = 15
    PURPLE_SAMPLE = 19
    GRAY_SAMPLE = 23
    UNUSED = 27
    ROW = 28
    USED = 29
    GHOST = 30
    COL = 31

    # Broad floor patches exclude thin decorative wiring.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            for color in (2, 5, 8, 9, 11, 14, 15):
                w1(FLOOR, color, dy, dx)
    p["b1"][FLOOR] = -24.0

    def player_template(channel, color, cy=0, cx=0):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                c = 5 if dy == 0 and dx == 0 else color
                w1(channel, c, cy + dy, cx + dx)
        p["b1"][channel] = -8.0

    player_template(PLAYER, 9)
    player_template(GHOST, 2)
    for i, (dy, dx) in enumerate(directions):
        player_template(SHIFTED_PLAYER + i, 9, -dy, -dx)

    # Exit's isolated orange center.
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            c = 9 if dy == 0 and dx == 0 else 5
            w1(EXIT, c, dy, dx)
    p["b1"][EXIT] = -8.0

    # White portal frames are not solid doors.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            w1(DOOR, 8, dy, dx)
            w1(DOOR, 11, dy, dx)
            w1(PURPLE_DOOR, 11, dy, dx)
    p["b1"][DOOR] = -11.0
    p["b1"][PURPLE_DOOR] = -11.0

    def switch_template(channel, colors):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                for c in colors:
                    w1(channel, c, dy, dx)
        for dy in (-2, 2):
            for dx in (-2, 2):
                w1(channel, 5, dy, dx)
        p["b1"][channel] = -12.0

    switch_template(RECORD_SWITCH, (8, 15))
    switch_template(PURPLE_SWITCH, (11,))

    for i, (dy, dx) in enumerate(directions):
        sy, sx = 2 * dy, 2 * dx
        py, px = -dx, dy
        for c in (8, 15):
            w1(RECORD_WIRE + i, c, sy, sx)
            w1(RECORD_WIRE + i, c, sy + py, sx + px, -0.5)
            w1(RECORD_WIRE + i, c, sy - py, sx - px, -0.5)
        w1(WHITE_SAMPLE + i, 15, sy, sx)
        w1(PURPLE_SAMPLE + i, 11, sy, sx)
        w1(GRAY_SAMPLE + i, 5, sy, sx)

    for dy in range(-1, 2):
        for dx in range(-1, 2):
            w1(UNUSED, 1, dy, dx)
            c = 0 if dy == 0 and dx == 0 else 2
            w1(USED, c, dy, dx)
    p["b1"][UNUSED] = -8.0
    p["b1"][USED] = -8.0

    w1(ROW, 16)
    w1(COL, 17)

    # ---------------------------------------------------------------
    # Second convolution: masks, switch contexts and counters.
    # ---------------------------------------------------------------
    F_FLOOR = 0
    F_BLOCK = 1
    F_EXIT = 2
    F_RECORD_SWITCH = 3
    F_PURPLE_SWITCH = 4
    F_DIRECTION = 5
    F_PLAYER = 9
    F_ON_RECORD_WIRE = 10
    F_PORTAL = 11
    F_CLOSED_PURPLE = 12
    F_UNUSED = 16
    F_PURPLE_BLOCK = 18
    F_GHOST = 19
    F_USED = 20
    F_PLAYER_ROW = 21
    F_PLAYER_COL = 22

    for out, inp in (
        (F_FLOOR, FLOOR),
        (F_EXIT, EXIT),
        (F_PLAYER, PLAYER),
        (F_UNUSED, UNUSED),
        (F_GHOST, GHOST),
        (F_USED, USED),
    ):
        w2(out, inp)

    for dy in range(-1, 2):
        for dx in range(-1, 2):
            w2(F_BLOCK, DOOR, dy, dx)
            w2(F_PURPLE_BLOCK, PURPLE_DOOR, dy, dx)
            w2(F_RECORD_SWITCH, RECORD_SWITCH, dy, dx)
            w2(F_PURPLE_SWITCH, PURPLE_SWITCH, dy, dx)

    for i, (dy, dx) in enumerate(directions):
        w2(F_DIRECTION + i, SHIFTED_PLAYER + i, -dy, -dx)

    w2(F_ON_RECORD_WIRE, PLAYER, value=4.0)
    for i, (dy, dx) in enumerate(directions):
        w2(F_ON_RECORD_WIRE, RECORD_WIRE + i, dy, dx)
    p["b2"][F_ON_RECORD_WIRE] = -4.0

    for i, (dy, dx) in enumerate(directions):
        w2(F_PORTAL, WHITE_SAMPLE + i, dy, dx)
    p["b2"][F_PORTAL] = -3.0

    # Tags adjacent to closed purple doors, aligned with their wires.
    for i, (py, px) in enumerate(directions):
        out = F_CLOSED_PURPLE + i
        w2(out, PURPLE_DOOR, -py, -px)
        w2(out, GRAY_SAMPLE + i, py, px, 4.0)
        ty, tx = -px, py
        for sign in (-1, 1):
            dy, dx = sign * ty, sign * tx
            j = direction_index[(dy, dx)]
            w2(out, PURPLE_SAMPLE + j, dy - py, dx - px, 2.0)
        p["b2"][out] = -19.0

    for out, inp in ((F_PLAYER_ROW, ROW), (F_PLAYER_COL, COL)):
        w2(out, PLAYER, value=2.0)
        w2(out, inp)
        p["b2"][out] = -2.0

    # ---------------------------------------------------------------
    # Axial layer.
    # ---------------------------------------------------------------
    S_FREE = 0
    S_EXIT = 1
    S_RECORD_SWITCH = 2
    S_PURPLE_SWITCH = 3
    S_PORTAL = 4
    S_DIRECTION = 5
    S_UNUSED = 9
    S_RECORD_POS = 10
    S_RECORD_EXCESS = 11
    S_ON_PORTAL = 12
    S_PURPLE_BLOCK = 13
    S_GHOST = 14
    S_USED = 15
    S_BASE = 16
    S_PLAYER_ROW = 18
    S_PLAYER_COL = 19

    p["W3"][S_FREE, F_FLOOR] = 1.0
    p["W3"][S_FREE, F_BLOCK] = -1.0
    p["W3"][S_BASE, F_FLOOR] = 1.0

    for out, inp in (
        (S_EXIT, F_EXIT),
        (S_RECORD_SWITCH, F_RECORD_SWITCH),
        (S_PORTAL, F_PORTAL),
        (S_UNUSED, F_UNUSED),
        (S_PURPLE_BLOCK, F_PURPLE_BLOCK),
        (S_GHOST, F_GHOST),
        (S_USED, F_USED),
        (S_PLAYER_ROW, F_PLAYER_ROW),
        (S_PLAYER_COL, F_PLAYER_COL),
    ):
        p["W3"][out, inp] = 1.0

    for i in range(4):
        p["W3"][S_DIRECTION + i, F_DIRECTION + i] = 1.0

    p["W3"][S_PURPLE_SWITCH, F_PURPLE_SWITCH] = 32.0
    p["b3"][S_PURPLE_SWITCH] = -32.0
    for i, (py, px) in enumerate(directions):
        axial_offset = 64 if px != 0 else 32
        p["W3"][S_PURPLE_SWITCH,
                axial_offset + F_CLOSED_PURPLE + i] = 1.0

    p["W3"][S_RECORD_POS, F_ON_RECORD_WIRE] = 1.0
    p["W3"][S_RECORD_EXCESS, F_ON_RECORD_WIRE] = 1.0
    p["b3"][S_RECORD_EXCESS] = -1.0

    p["W3"][S_ON_PORTAL, F_PORTAL] = 1.0
    p["W3"][S_ON_PORTAL, F_PLAYER] = 1.0
    p["b3"][S_ON_PORTAL] = -1.0

    # ---------------------------------------------------------------
    # Recurrent spatial computation.
    #
    # Each pair represents a clipped flood as h[a] - h[a+1].
    # Directional accumulators integrate arrival times.
    # ---------------------------------------------------------------
    obstacle_penalty = 64.0

    def flood_pair(first, free_channel, sources):
        for out in (first, first + 1):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    wh(out, first, dy, dx, 1.0)
                    wh(out, first + 1, dy, dx, -1.0)
            p["Uh"][out, free_channel] = obstacle_penalty
            for source, weight in sources:
                p["Uh"][out, source] += weight
        p["bh"][first] = -obstacle_penalty
        p["bh"][first + 1] = -obstacle_penalty - 1.0

    flood_pair(0, S_FREE, ((S_EXIT, 1.0),))
    flood_pair(
        2, S_FREE,
        ((S_RECORD_SWITCH, 1.0), (S_PURPLE_SWITCH, 1.0))
    )

    # Relaxed portal routes and closed-purple-door approach targets.
    # Actual movement selection remains gated by the strict free mask.
    flood_pair(
        4, S_BASE,
        ((S_PORTAL, 1.0), (S_PURPLE_BLOCK, obstacle_penalty))
    )

    FREE_H = 14
    p["Uh"][FREE_H, S_FREE] = 1.0

    local_gate = 1048576.0
    exit_priority = 4096.0
    switch_priority = 16.0

    for policy in range(2):
        for i, (dy, dx) in enumerate(directions):
            out = 6 + 4 * policy + i
            wh(out, out)

            wh(out, 0, dy, dx, exit_priority)
            wh(out, 1, dy, dx, -exit_priority)

            if policy == 0:
                wh(out, 2, dy, dx, switch_priority)
                wh(out, 3, dy, dx, -switch_priority)

            wh(out, 4, dy, dx, 1.0)
            wh(out, 5, dy, dx, -1.0)

            wh(out, FREE_H, dy, dx, local_gate)
            p["Uh"][out, S_DIRECTION + i] = local_gate
            p["bh"][out] = 1.0 - 2.0 * local_gate

    # ---------------------------------------------------------------
    # Pooled policy selection.
    # ---------------------------------------------------------------
    def fmax(channel):
        return 32 + channel

    def fmean(channel):
        return 64 + channel

    policy_gate = 1048576.0

    for i in range(4):
        # Ordinary policy: exits, switches, then approach targets.
        p["Wz"][i, 6 + i] = 1.0
        p["Wz"][i, fmax(S_UNUSED)] = policy_gate
        p["Wz"][i, fmax(S_PORTAL)] = -policy_gate

        # Final-recording portal policy.
        z = 4 + i
        p["Wz"][z, 10 + i] = 1.0
        p["Wz"][z, fmax(S_UNUSED)] = -policy_gate
        p["Wz"][z, fmax(S_PORTAL)] = policy_gate
        p["bz"][z] = -policy_gate

        p["Wo"][1 + i, i] = 1.0
        p["Wo"][1 + i, 4 + i] = 1.0

    COMMIT_Z = 8
    p["Wz"][COMMIT_Z, fmax(S_RECORD_POS)] = 1.0
    p["Wz"][COMMIT_Z, fmax(S_RECORD_EXCESS)] = -1.0
    p["Wz"][COMMIT_Z, fmax(S_UNUSED)] = 1.0
    p["Wz"][COMMIT_Z, fmax(S_ON_PORTAL)] = -1.0
    p["Wz"][COMMIT_Z, fmean(S_GHOST)] = 4096.0
    p["Wz"][COMMIT_Z, fmean(S_USED)] = -4096.0
    p["bz"][COMMIT_Z] = -1.0

    # ---------------------------------------------------------------
    # Recording-position memory.
    #
    # Four bits per axis encode four-pixel bins.
    #   0:8   current coordinate bits
    #   8:24  positive/negative write gates
    #  24:32  latched coordinate bits
    #  32:48  current-versus-latched comparison units
    # ---------------------------------------------------------------
    ramps = []
    for axis, feature in enumerate((S_PLAYER_ROW, S_PLAYER_COL)):
        axis_ramps = []
        for j in range(1, 16):
            boundary = 4.0 * j - 0.5
            lo = 10 + axis * 30 + 2 * (j - 1)
            hi = lo + 1
            p["Wz"][lo, fmax(feature)] = 63.0
            p["Wz"][hi, fmax(feature)] = 63.0
            p["bz"][lo] = -(boundary - 0.25)
            p["bz"][hi] = -(boundary + 0.25)
            axis_ramps.append((lo, hi))
        ramps.append(axis_ramps)

    bit_coefficients = []
    for axis in range(2):
        for bit in range(4):
            coeff = {}
            stride = 1 << bit
            for j in range(stride, 16, stride):
                sign = 1.0 if (j // stride) % 2 else -1.0
                lo, hi = ramps[axis][j - 1]
                coeff[lo] = 4.0 * sign
                coeff[hi] = -4.0 * sign
            bit_coefficients.append(coeff)

    def add_current_bit(mem_unit, bit, scale):
        p["bm"][mem_unit] -= scale
        for z, weight in bit_coefficients[bit].items():
            p["Wmz"][mem_unit, z] += scale * weight

    for bit in range(8):
        current = bit
        write_pos = 8 + 2 * bit
        write_neg = write_pos + 1
        latch = 24 + bit
        compare_pos = 32 + 2 * bit
        compare_neg = compare_pos + 1

        add_current_bit(current, bit, 12.0)

        # Capture the preceding position when ACTION5 was issued.
        p["Wm"][write_pos, current] = 12.0
        p["Wm"][write_neg, current] = -12.0
        p["Wma"][write_pos, 5] = 24.0
        p["Wma"][write_neg, 5] = 24.0
        p["bm"][write_pos] = -24.0
        p["bm"][write_neg] = -24.0

        p["Wm"][latch, latch] = 12.0
        p["Wm"][latch, write_pos] = 24.0
        p["Wm"][latch, write_neg] = -24.0

        # Matching bits give comparator sum zero; mismatches give -2.
        add_current_bit(compare_pos, bit, 12.0)
        add_current_bit(compare_neg, bit, -12.0)
        p["Wm"][compare_pos, latch] = 12.0
        p["Wm"][compare_neg, latch] = -12.0
        p["bm"][compare_pos] -= 12.0
        p["bm"][compare_neg] -= 12.0

        # Account for a pending latch write.
        p["Wm"][compare_pos, write_pos] = 24.0
        p["Wm"][compare_pos, write_neg] = -24.0
        p["Wm"][compare_neg, write_pos] = -24.0
        p["Wm"][compare_neg, write_neg] = 24.0

    mismatch_scale = 4.0 * policy_gate
    commit_scale = 16.0 * mismatch_scale
    p["Wo"][5, COMMIT_Z] = commit_scale
    p["bo"][5] = -commit_scale - mismatch_scale
    for unit in range(32, 48):
        p["Wo"][5, 128 + unit] = -mismatch_scale

    # ---------------------------------------------------------------
    # Directional persistence, disabled for portal layouts.
    # ---------------------------------------------------------------
    PORTAL_PRESENT_Z = 70
    p["Wz"][PORTAL_PRESENT_Z, fmax(S_PORTAL)] = 1.0

    for i in range(4):
        unit = 48 + i
        action = 1 + i
        p["Wma"][unit, action] = 12.0
        p["Wmz"][unit, PORTAL_PRESENT_Z] = -24.0
        p["bm"][unit] = -6.0
        p["Wo"][action, 128 + unit] = 8.0

    # ---------------------------------------------------------------
    # One-time episode normalization.
    #
    # A constant and its delayed copy produce a pulse only on the
    # second invocation. Taking an action before RESET also avoids
    # the engine's zero-action-count full-reset case.
    # ---------------------------------------------------------------
    STARTED = 52
    DELAYED_STARTED = 53
    RESET_PULSE = 54

    p["bm"][STARTED] = 12.0
    p["Wm"][DELAYED_STARTED, STARTED] = 12.0

    p["Wm"][RESET_PULSE, STARTED] = 12.0
    p["Wm"][RESET_PULSE, DELAYED_STARTED] = -24.0
    p["bm"][RESET_PULSE] = -6.0

    reset_strength = float(2 ** 30)
    p["Wo"][0, 128 + RESET_PULSE] = reset_strength

    # Do not create a recording before normalization. This also
    # leaves the recording-position latch clean across the reset.
    initial_commit_gate = float(2 ** 29)
    p["Wo"][5, 128 + DELAYED_STARTED] = initial_commit_gate
    p["bo"][5] -= initial_commit_gate

    p["bo"][6] = -1.0e9
    p["bo"][7] = -1.0e9

    return p
