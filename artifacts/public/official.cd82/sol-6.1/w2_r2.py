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
    p = {k: np.zeros(shape, np.float32) for k, shape in shapes.items()}
    W1, b1 = p["W1"], p["b1"]
    W2, b2 = p["W2"], p["b2"]
    W3, b3 = p["W3"], p["b3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]
    Wz, bz = p["Wz"], p["bz"]

    # Integer coordinates and shared coordinate hinges.
    W1[0, 16, 2, 2] = 63
    W1[1, 17, 2, 2] = 63

    row_hinge = {}
    for ch, r in enumerate([4, 7, 11, 35, 38, 42], start=2):
        row_hinge[r] = ch
        W1[ch, 16, 2, 2] = 63
        b1[ch] = -r

    col_hinge = {}
    for ch, c in enumerate([6, 9, 30, 33], start=8):
        col_hinge[c] = ch
        W1[ch, 17, 2, 2] = 63
        b1[ch] = -c

    BG, UPPER_RIGHT, NUM = 12, 13, 14
    PAL_PATTERN, BLACK_BELOW, MINI_PATTERN, RED = 15, 16, 17, 18
    POSE_HINGE = 19

    for colour in range(16):
        if colour in {2, 3, 4, 5}:
            W1[BG, colour, 2, 2] = 1
        else:
            W1[NUM, colour, 2, 2] = colour + 1

    W1[UPPER_RIGHT, 17, 2, 2] = 63
    W1[UPPER_RIGHT, 16, 2, 2] = -63
    b1[UPPER_RIGHT] = -10

    for dy, dx in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        W1[PAL_PATTERN, 4, 2 + dy, 2 + dx] = 1
    b1[PAL_PATTERN] = -3

    W1[BLACK_BELOW, 0, 4, 2] = 1
    W1[RED, 2, 2, 2] = 1

    # A small-tool eye is black with four nearby cardinal red neighbours.
    W1[MINI_PATTERN, 0, 2, 2] = 10
    for distance in [1, 2]:
        for dy, dx in [
            (-distance, 0), (distance, 0),
            (0, -distance), (0, distance),
        ]:
            W1[MINI_PATTERN, 2, 2 + dy, 2 + dx] = 1
    b1[MINI_PATTERN] = -13

    # Fixed basket geometry, not level-specific drawing instructions.
    landmarks = [
        (24, 30), (21, 39), (37, 46), (46, 49),
        (53, 30), (56, 24), (37, 17), (31, 14),
    ]
    for i, (r, c) in enumerate(landmarks):
        value = 64 * r + c
        W1[POSE_HINGE + i, 16, 2, 2] = 63 * 64
        W1[POSE_HINGE + i, 17, 2, 2] = 63
        b1[POSE_HINGE + i] = -value

    # Row hats. Neighbouring rows share a first-layer hinge.
    rows = [4, 7, 8, 11, 35, 38, 39, 42]
    row_hat = {}
    for ch, r in enumerate(rows):
        base = {8: 7, 39: 38}.get(r, r)
        dy = base - r
        row_hat[r] = ch
        W2[ch, 0, 1 + dy, 1] = 1
        W2[ch, row_hinge[base], 1 + dy, 1] = -2
        b2[ch] = 1 - base

    # Column-gated numeric buses. On the selected column they contain
    # 100 + colour + 1; elsewhere they are zero.
    goal_columns = [5, 6, 7, 9, 10]
    canvas_columns = [c + 24 for c in goal_columns]
    column_bus = {}
    for ch, c in enumerate(goal_columns + canvas_columns, start=8):
        if c in [5, 6, 7]:
            base = 6
        elif c in [9, 10]:
            base = 9
        elif c in [29, 30, 31]:
            base = 30
        else:
            base = 33
        dx = base - c
        column_bus[c] = ch
        W2[ch, NUM, 1, 1] = 1
        W2[ch, UPPER_RIGHT, 1, 1] = -100
        W2[ch, 1, 1, 1 + dx] += 200
        W2[ch, col_hinge[base], 1, 1 + dx] -= 400
        b2[ch] = 200 * (1 - base) - 100

    POSE2, PAL_CODE, SELECTED_CODE = 18, 26, 27
    MINI2, BG2, RAW_ROW = 28, 29, 30

    for i, (r, c) in enumerate(landmarks):
        value = 64 * r + c
        W2[POSE2 + i, 0, 1, 1] = 4 * 64
        W2[POSE2 + i, 1, 1, 1] = 4
        W2[POSE2 + i, POSE_HINGE + i, 1, 1] = -8
        W2[POSE2 + i, RED, 1, 1] = 100
        b2[POSE2 + i] = 1 - 4 * value - 100

    W2[PAL_CODE, NUM, 1, 1] = 1
    W2[PAL_CODE, PAL_PATTERN, 1, 1] = 20
    b2[PAL_CODE] = -20

    W2[SELECTED_CODE, NUM, 1, 1] = 1
    W2[SELECTED_CODE, PAL_PATTERN, 1, 1] = 20
    W2[SELECTED_CODE, BLACK_BELOW, 2, 1] = 20
    b2[SELECTED_CODE] = -40

    W2[MINI2, MINI_PATTERN, 1, 1] = 1
    W2[BG2, BG, 1, 1] = 1
    W2[RAW_ROW, 0, 1, 1] = 1

    # Eight unpatched sectors followed by four stamp witnesses.
    # Every sample is off both ignored diagonals.
    samples = [
        (1, 2), (1, 7),
        (4, 3), (5, 3),
        (4, 6), (5, 6),
        (8, 2), (8, 7),
        (1, 4), (4, 2), (4, 7), (8, 4),
    ]
    sample_index = {site: i for i, site in enumerate(samples)}
    n = len(samples)

    FPOSE, FSOURCE, FPAL, FSELECTED, FMINI = 12, 20, 28, 29, 30

    # Each comparison channel has two nonzero sites:
    # target_colour + 1 and canvas_colour + 32.
    for i, (r, c) in enumerate(samples):
        gr, gc = r + 3, c + 3
        cr, cc = r + 34, c + 27
        W3[i, column_bus[gc]] = 1
        W3[i, column_bus[cc]] = 1
        W3[i, row_hat[gr]] = 100
        W3[i, row_hat[cr]] = 100
        W3[i, RAW_ROW] = 1
        W3[i, BG2] = -100
        b3[i] = -gr - 200

    for i in range(8):
        W3[FPOSE + i, 64 + POSE2 + i] = 1

    desired_sites = [
        (1, 4), (1, 7), (4, 7), (4, 6),
        (8, 4), (8, 2), (4, 2), (4, 3),
    ]
    source_gate = 20.0
    for pose, (r, c) in enumerate(desired_sites):
        gr, gc = r + 3, c + 3
        W3[FSOURCE + pose, 32 + column_bus[gc]] = 1
        W3[FSOURCE + pose, row_hat[gr]] = source_gate
        W3[FSOURCE + pose, 64 + POSE2 + pose] = source_gate
        b3[FSOURCE + pose] = -100 - 2 * source_gate

    W3[FPAL, PAL_CODE] = 1
    W3[FSELECTED, SELECTED_CODE] = 1
    W3[FMINI, MINI2] = 1

    desired_source = np.zeros(32, np.float32)
    desired_source[FSOURCE:FSOURCE + 8] = 1
    selected_source = np.zeros(32, np.float32)
    selected_source[FSELECTED] = 1

    # Separable prefix sums broadcast sparse desired and selected colours.
    def broadcast_block(start, source):
        left, right, up, down = range(start, start + 4)
        Wh[left, left, 1, 2] = 1
        Wh[right, right, 1, 0] = 1
        Uh[left] = source
        Uh[right] = source
        for out, dy in [(up, 2), (down, 0)]:
            Wh[out, out, dy, 1] = 1
            Wh[out, left, 1, 1] = 1
            Wh[out, right, 1, 1] = 1
            Uh[out] = -source

    broadcast_block(0, desired_source)
    broadcast_block(4, selected_source)
    global_coeff = np.array([-1, -1, 1, 1], np.float32)
    K = 100.0

    # Colour mismatch penalties at small-tool eyes.
    for out, sign in [(8, 1.0), (9, -1.0)]:
        Wh[out, 0:4, 1, 1] = sign * global_coeff
        Wh[out, 4:8, 1, 1] = -sign * global_coeff
        Uh[out] = sign * (desired_source - selected_source)
        Uh[out, FMINI] += K
        bh[out] = -K

    # Palette indicator and palette-colour distance.
    Uh[12, FPAL] = 1
    bh[12] = -1
    Uh[13, FPAL] = 1
    Wh[13, 12, 1, 1] = -1

    for out, sign in [(10, 1.0), (11, -1.0)]:
        Wh[out, 0:4, 1, 1] = sign * global_coeff
        Wh[out, 13, 1, 1] = K
        Uh[out] = sign * desired_source
        Uh[out, FPAL] -= sign
        bh[out] = -K

    p["Wc"][0, FPAL] = 20
    p["Wc"][0, FMINI] = 40
    p["Wc"][0, 32 + 12] = -20
    p["Wc"][0, 32 + 8] = -100
    p["Wc"][0, 32 + 9] = -100
    p["Wc"][0, 32 + 10] = -2
    p["Wc"][0, 32 + 11] = -2

    def fm(ch):
        return 32 + ch

    def fa(ch):
        return 64 + ch

    # Recover target from mean and canvas from max, then compare.
    for i in range(n):
        for unit, sign in [(2 * i, 1.0), (2 * i + 1, -1.0)]:
            Wz[unit, fa(i)] = sign * 4096
            Wz[unit, fm(i)] = -sign * 2
            bz[unit] = sign * 31 - 0.03

        for unit, sign in [
            (24 + 2 * i, 1.0), (25 + 2 * i, -1.0),
        ]:
            Wz[unit, fa(i)] = sign * 4096
            Wz[unit, fm(i)] = -sign
            Wz[unit, fm(FSELECTED)] = -sign
            bz[unit] = -0.03

        # Target-versus-desired guards for witness strokes.
        for unit, sign in [
            (60 + 2 * i, 1.0), (61 + 2 * i, -1.0),
        ]:
            Wz[unit, fa(i)] = sign * 4096
            Wz[unit, fm(i)] = -sign
            for pose in range(8):
                Wz[unit, fm(FSOURCE + pose)] -= sign
            bz[unit] = -0.1

    SELECTED_ERROR, POSE_Z, MINI_Z = 48, 50, 58
    for unit, sign in [(48, 1.0), (49, -1.0)]:
        Wz[unit, fm(FSELECTED)] = sign
        for pose in range(8):
            Wz[unit, fm(FSOURCE + pose)] -= sign
        bz[unit] = -0.1

    for pose in range(8):
        Wz[POSE_Z + pose, fm(FPOSE + pose)] = 1
    Wz[MINI_Z, fm(FMINI)] = 1

    def in_stroke(pose, r, c):
        if pose == 0:
            return r < 5
        if pose == 1:
            return c > r
        if pose == 2:
            return c >= 5
        if pose == 3:
            return r + c > 9
        if pose == 4:
            return r >= 5
        if pose == 5:
            return c < r
        if pose == 6:
            return c < 5
        return r + c < 9

    M, bm = p["Wmz"], p["bm"]
    sharp, gate = 10.0, 200.0

    def selected_guard(unit):
        M[unit, SELECTED_ERROR:SELECTED_ERROR + 2] -= sharp * gate

    def target_guard(unit, i):
        M[unit, 60 + 2 * i:62 + 2 * i] -= sharp * gate

    cardinal_bases = {
        0: [0, 1],
        2: [4, 5],
        4: [6, 7],
        6: [2, 3],
    }

    # Greedy sampled-L1 improvement predicates.
    for pose in range(8):
        for i, (r, c) in enumerate(samples):
            if in_stroke(pose, r, c):
                M[pose, 2 * i:2 * i + 2] += sharp
                M[pose, 24 + 2 * i:26 + 2 * i] -= sharp
        M[pose, POSE_Z + pose] += sharp * gate
        selected_guard(pose)
        bm[pose] = sharp * (-gate - 0.4)
        for i in cardinal_bases.get(pose, []):
            target_guard(pose, i)

    # On stamp levels, right/left broad strokes are not selected merely
    # because painting over unfinished patches gives a transient gain.
    M[2, MINI_Z] -= sharp * gate
    M[6, MINI_Z] -= sharp * gate

    # Necessary underlay strokes, guarded by a matching-colour witness.
    witnesses = [
        (1, 2), (1, 7), (8, 7), (4, 6),
        (8, 2), (8, 2), None, (4, 3),
    ]
    for pose, site in enumerate(witnesses):
        unit = 8 + pose
        if site is None:
            bm[unit] = -100
            continue
        i = sample_index[site]
        M[unit, 2 * i:2 * i + 2] = sharp
        M[unit, POSE_Z + pose] = sharp * gate
        selected_guard(unit)
        target_guard(unit, i)
        bm[unit] = sharp * (-gate - 0.6)
        for j in cardinal_bases.get(pose, []):
            target_guard(unit, j)

    # A northwest restoration may be needed even after its left witness
    # is correct: a later right-half stroke can spoil the upper-right base.
    unit = 16
    i = sample_index[(1, 7)]
    M[unit, 2 * i:2 * i + 2] = sharp
    M[unit, POSE_Z + 7] = sharp * gate
    selected_guard(unit)
    target_guard(unit, i)
    bm[unit] = sharp * (-gate - 0.6)

    # Colour-change request.
    M[17, SELECTED_ERROR:SELECTED_ERROR + 2] = sharp
    bm[17] = -0.5 * sharp

    # Stamp requests use their own off-diagonal comparison witnesses.
    patch_sites = [(1, 4), (4, 7), (8, 4), (4, 2)]
    for j, (pose, site) in enumerate(zip([0, 2, 4, 6], patch_sites)):
        unit = 18 + j
        i = sample_index[site]
        M[unit, 2 * i:2 * i + 2] = sharp
        M[unit, POSE_Z + pose] = sharp * gate
        M[unit, MINI_Z] = sharp * gate
        selected_guard(unit)
        bm[unit] = sharp * (-2 * gate - 0.6)

    # Scan the eight-position perimeter when no drawing request is active.
    clockwise_action = [4, 2, 2, 3, 3, 1, 1, 4]
    p["bo"][:] = -100
    for action in [1, 2, 3, 4]:
        p["bo"][action] = -1
    for pose, action in enumerate(clockwise_action):
        p["Wo"][action, POSE_Z + pose] = 1

    # Inactive predicates are -1. Baselines keep both drawing actions
    # below movement unless at least one predicate is active.
    p["Wo"][5, 128:145] = 1
    p["bo"][5] = 16
    p["Wo"][6, 145:150] = 1
    p["bo"][6] = 4.2

    return p
