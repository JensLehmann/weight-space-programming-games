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
    p = {k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()}

    templates = [
        ["00100", "00100", "01110", "00100", "11111"],
        ["00001", "00101", "11111", "10100", "10000"],
        ["11111", "10100", "10000", "10100", "11111"],
        ["00100", "11111", "10101", "10101", "00100"],
        ["10001", "11111", "00100", "11111", "10001"],
        ["11111", "00100", "11100", "10000", "11111"],
        ["10001", "11111", "10001", "10001", "11011"],

        ["10000", "11110", "10010", "10010", "11111"],
        ["11111", "10001", "11101", "10101", "11111"],
        ["00111", "00101", "11111", "10100", "11100"],
        ["01110", "01010", "11111", "10001", "11111"],
        ["11111", "10001", "10001", "11111", "00100"],
        ["11110", "10011", "10001", "11001", "01111"],
        ["00100", "11111", "10101", "11111", "00100"],

        ["10101", "10000", "10101", "10000", "11111"],
        ["11101", "10000", "00000", "00001", "10111"],
        ["10111", "00100", "00100", "00100", "11101"],
        ["00100", "10101", "00100", "10101", "00100"],
        ["11011", "10001", "10101", "10001", "11011"],
        ["10111", "10101", "11101", "00000", "10101"],
        ["10001", "00000", "10001", "11011", "01010"],
    ]
    glyphs = np.asarray(
        [[[int(c) for c in row] for row in t] for t in templates],
        dtype=np.int32,
    )

    orbits = [
        [(2, 2)],
        [(1, 2), (2, 3), (3, 2), (2, 1)],
        [(1, 1), (1, 3), (3, 3), (3, 1)],
        [(0, 2), (2, 4), (4, 2), (2, 0)],
        [(0, 0), (0, 4), (4, 4), (4, 0)],
        [(0, 1), (1, 4), (4, 3), (3, 0)],
        [(0, 3), (3, 4), (4, 1), (1, 0)],
    ]
    signatures = np.asarray([
        [sum(int(g[y, x]) for y, x in orbit) for orbit in orbits]
        for g in glyphs
    ], dtype=np.int32)

    # Analytic hinge basis for rotation-invariant signature distances.
    hinge_channel = {}
    channel = 0
    for o, orbit in enumerate(orbits):
        n = len(orbit)
        thresholds = [0] + sorted({
            int(v) for v in signatures[:, o] if 0 < int(v) < n
        })
        for threshold in thresholds:
            hinge_channel[o, threshold] = channel
            for y, x in orbit:
                p["W1"][channel, 5, y, x] = 1.0
            p["b1"][channel] = -float(threshold)
            channel += 1
    assert channel == 21

    # Non-tile-colour counters, shifted in W2 onto the 7x7 border.
    tile_colours = {7, 10, 11}
    edges = [
        [(0, x) for x in range(5)],
        [(4, x) for x in range(5)],
        [(y, 0) for y in range(5)],
        [(y, 4) for y in range(5)],
    ]
    for k, edge in enumerate(edges):
        for colour in range(16):
            if colour not in tile_colours:
                for y, x in edge:
                    p["W1"][21 + k, colour, y, x] = 1.0

    # Coordinates and vertical hinges.
    p["W1"][25, 16, 2, 2] = 63.0
    p["W1"][26, 17, 2, 2] = 63.0
    p["W1"][27, 16, 2, 2] = 63.0
    p["b1"][27] = -34.0
    p["W1"][28, 16, 2, 2] = 63.0
    p["b1"][28] = -48.0

    # Upper cursor bracket at the first selected glyph's centre.
    for x in range(5):
        p["W1"][29, 0, 2, x] = 1.0
    p["W1"][29, 0, 3, 0] = 1.0
    p["W1"][29, 0, 3, 1] = -1.0
    p["b1"][29] = -5.0

    # Row-family signal: C contributes 1, B contributes 2.
    p["W1"][30, 11, 0, 2] = 1.0
    p["W1"][30, 7, 0, 2] = 2.0

    edge_samples = [(0, 1), (2, 1), (1, 0), (1, 2)]
    for g in range(21):
        bias = 1.0
        for o, orbit in enumerate(orbits):
            target = int(signatures[g, o])
            n = len(orbit)
            c0 = hinge_channel[o, 0]

            if target == n:
                p["W2"][g, c0, 1, 1] += 2.0
                bias -= 2.0 * n
            else:
                p["W2"][g, c0, 1, 1] += 2.0
                p["W2"][g, hinge_channel[o, target], 1, 1] -= 4.0
                bias -= 2.0 * target

        for k, (y, x) in enumerate(edge_samples):
            p["W2"][g, 21 + k, y, x] = -2.0
        p["b2"][g] = bias

    p["W2"][21, 29, 1, 1] = 1.0
    p["W2"][22, 25, 1, 1] = 1.0
    p["W2"][23, 26, 1, 1] = 1.0

    for out, threshold in [(24, 14.5), (25, 21.5)]:
        p["W2"][out, 25, 1, 1] = 1.0
        p["b2"][out] = -threshold

    p["W2"][26, 27, 1, 1] = 1.0
    p["W2"][27, 28, 1, 1] = 1.0

    # Source and target partitions have different horizontal boundaries.
    p["W2"][28, 26, 1, 1] = 1.0
    p["b2"][28] = -26.5
    p["W2"][31, 26, 1, 1] = 1.0
    p["b2"][31] = -36.5

    # Margin prevents coordinate roundoff from reducing glyph counts.
    p["W2"][29, 25, 1, 1] = -1.0
    p["W2"][29, 27, 1, 1] = 1.0
    p["W2"][29, 28, 1, 1] = 1.0
    p["b2"][29] = 33.75

    p["W2"][30, 30, 0, 1] = 1.0

    C = 16384.0
    family_gate = 65536.0
    table_gate = 65536.0

    # Input presence, input code, and ungated source-family code.
    for family in (0, 2):
        for j in range(7):
            p["W3"][0, 7 * family + j] = 1.0
            p["W3"][1, 7 * family + j] = j + 1
            p["W3"][3, 7 * family + j] = j + 1
    p["W3"][0, 29] = -512.0
    p["W3"][1, 29] = -512.0

    # Current lower output code.
    for family in range(3):
        for j in range(7):
            p["W3"][2, 7 * family + j] = j + 1
    p["W3"][2, 64 + 21] = 8.0
    p["W3"][2, 22] = 512.0
    p["W3"][2, 27] = -512.0
    p["b3"][2] = -8.0 - 48.0 * 512.0

    def region_penalty(row, col, side):
        v = np.zeros(32, dtype=np.float32)
        bias = 0.0
        if row == 0:
            v[24] += 1.0
        elif row == 1:
            bias += 14.5
            v[22] -= 1.0
            v[24] += 1.0
            v[25] += 1.0
        else:
            bias += 21.5
            v[22] -= 1.0
            v[25] += 1.0
            v[26] += 1.0

        hinge = 28 if side == 0 else 31
        threshold = 26.5 if side == 0 else 36.5
        if col == 0:
            v[hinge] += 1.0
        else:
            bias += threshold
            v[23] -= 1.0
            v[hinge] += 1.0
        return v, bias

    # Four fields per rule: source -x/+x and target -x/+x.
    for r in range(6):
        row, col = divmod(r, 2)
        for side in range(2):
            penalty, penalty_bias = region_penalty(row, col, side)
            for sign_index, sign in enumerate((-1.0, 1.0)):
                out = 4 + 4 * r + 2 * side + sign_index
                p["W3"][out, :32] -= table_gate * penalty
                p["W3"][out, 23] += sign
                p["b3"][out] = -table_gate * penalty_bias

                if side == 0:
                    p["b3"][out] -= 3.0 * family_gate
                    p["W3"][out, 32 + 30] = family_gate
                    for j in range(7):
                        p["W3"][out, j] += (
                            family_gate + C + j + 1
                        )
                        p["W3"][out, 14 + j] += (
                            2.0 * family_gate + C + j + 1
                        )
                else:
                    p["b3"][out] -= family_gate
                    p["W3"][out, 32 + 30] = -family_gate
                    for j in range(7):
                        p["W3"][out, j] += (
                            2.0 * family_gate + C + j + 1
                        )
                        p["W3"][out, 7 + j] += (
                            3.0 * family_gate + C + j + 1
                        )

    # Cursor-aligned input code for A->B single-symbol translation.
    for j in range(7):
        p["W3"][28, j] = j + 1
    p["W3"][28, 64 + 21] = 8.0
    p["W3"][28, 32 + 30] = -8.0
    p["W3"][28, 29] = -512.0
    p["b3"][28] = -8.0

    # C-input mode flag, restricted to the lower input strip.
    p["W3"][29, 32 + 30] = 1.0
    p["W3"][29, 29] = -512.0

    # Spatial prefix count along the lower input word.
    p["Wh"][0, 0, 1, 0] = 1.0
    p["Uh"][0, 0] = 1.0

    # Prefix tails whose pooled differences recover successive digits.
    M = 16.0
    blank_gate = 512.0
    for k in range(1, 9):
        p["Wh"][k, 0, 1, 0] = M
        p["Uh"][k, 0] = blank_gate + M * (1.5 - k)
        p["Uh"][k, 1] = 1.0
        p["bh"][k] = -blank_gate

    # Source positional baselines, with glyph codes subtracted.
    for r in range(6):
        p["Uh"][9 + r, 4 + 4 * r] = 1.0
        p["Uh"][9 + r, 3] = -1.0

    def fmax(c):
        return 32 + c

    def fmean(c):
        return 64 + c

    def hmean(c):
        return 16 + c

    z_used = 0

    def readout(vector, bias=0.0):
        nonlocal z_used
        z = z_used
        z_used += 1
        p["Wz"][z] = vector
        p["bz"][z] = bias
        return z

    z_digits = []
    for k in range(1, 9):
        v = np.zeros(96, dtype=np.float32)
        v[hmean(k)] = 4096.0
        if k < 8:
            v[hmean(k + 1)] = -4096.0
            v[fmean(0)] = -M * 4096.0
            bias = M * (k - 0.5)
        else:
            bias = -0.5 * M
        z_digits.append(readout(v, bias))

    v = np.zeros(96, dtype=np.float32)
    v[fmax(2)] = 1.0
    z_output = readout(v)

    v = np.zeros(96, dtype=np.float32)
    v[fmax(28)] = 1.0
    z_aligned_input = readout(v)

    v = np.zeros(96, dtype=np.float32)
    v[fmax(29)] = 1.0
    z_c_mode = readout(v)

    z_rules = []
    for r in range(6):
        lm, lp, rm, rp = [4 + 4 * r + t for t in range(4)]
        baseline = 9 + r

        lhs_first = np.zeros(96, dtype=np.float32)
        lhs_first[fmax(lm)] = 1.0
        lhs_first[baseline] = -1.0

        # Cancel position and glyph index to estimate source length.
        nl = np.zeros(96, dtype=np.float32)
        nl[fmean(lp)] = 4096.0 / (2.0 * C)
        nl[fmean(lm)] = -4096.0 / (2.0 * C)
        nl[hmean(baseline)] = 4096.0 / C

        nr = np.zeros(96, dtype=np.float32)
        nr[fmean(rm)] = 4096.0 / (2.0 * C)
        nr[fmean(rp)] = 4096.0 / (2.0 * C)

        rhs_first = 7.0 * nl
        rhs_first[fmax(rm)] += 1.0
        rhs_first[baseline] -= 1.0

        rhs_last = -7.0 * (nl + nr)
        rhs_last[fmax(rp)] += 1.0
        rhs_last[baseline] += 1.0

        z_rules.append((
            readout(lhs_first),
            readout(rhs_first, 3.0),
            readout(rhs_last, 4.0 - 2.0 * C),
            readout(nl),
            readout(nr),
        ))

    # Memory layout.
    PTR = 0
    BOUNDS = list(range(1, 7))
    FAR_POS = 7
    INPUT = list(range(8, 15))
    FAR_NEG = 15
    CMP = [(16 + 2 * r, 17 + 2 * r) for r in range(6)]
    TARGET = [(28 + 2 * r, 29 + 2 * r) for r in range(6)]
    LENGTH = list(range(40, 46))
    OUT_GE, OUT_LE = 46, 47
    RHS_TWO = 48
    PHASE, CONSUME, SET_PHASE = 49, 50, 51
    DONE, ADVANCE = 52, 53
    NEG_ONE, INIT = 54, 55
    HISTORY = list(range(56, 61))
    SET_DELAY = 61
    COOL, ALTERNATE = 62, 63

    ep, ei, et = 0.001, 0.01, 0.002
    gain = 50.0

    # Input-position integrator. Inactive event registers cancel exactly.
    p["Wm"][PTR, PTR] = 1.0
    p["Wm"][PTR, ADVANCE] = 1.0
    p["Wm"][PTR, CONSUME] = -0.5
    p["Wm"][PTR, NEG_ONE] = -0.5

    for i, u in enumerate(BOUNDS):
        p["Wm"][u, PTR] = 100.0 / ep
        p["bm"][u] = -100.0 * (i + 0.5)

    # C mode selects a prefix-indexed input digit.
    # A mode keeps these registers inactive and uses aligned input instead.
    for i, u in enumerate(INPUT):
        p["Wmz"][u, z_digits[i]] = ei
        p["Wmz"][u, z_c_mode] = 20.0
        p["bm"][u] = -20.0

        if i == 0:
            p["Wm"][u, BOUNDS[0]] = -10.0
            p["bm"][u] -= 10.0
        elif i == len(INPUT) - 1:
            p["Wm"][u, BOUNDS[-1]] = 10.0
            p["bm"][u] -= 10.0
        else:
            p["Wm"][u, BOUNDS[i - 1]] = 10.0
            p["Wm"][u, BOUNDS[i]] = -10.0
            p["bm"][u] -= 20.0

    # Equality to each rule's first source glyph.
    for r, (ge, le) in enumerate(CMP):
        zl, _, _, _, _ = z_rules[r]
        for u in INPUT:
            p["Wm"][ge, u] = gain / ei
            p["Wm"][le, u] = -gain / ei

        p["Wmz"][ge, zl] = -gain
        p["Wmz"][le, zl] = gain
        p["Wmz"][ge, z_aligned_input] = gain
        p["Wmz"][le, z_aligned_input] = -gain
        p["Wmz"][ge, z_c_mode] = -gain / ei
        p["Wmz"][le, z_c_mode] = gain / ei
        p["bm"][ge] = gain * (7.0 / ei + 0.2)
        p["bm"][le] = gain * (-7.0 / ei + 0.2)

    # Target registers contain only glyph codes, not length tags.
    for r, (first, last) in enumerate(TARGET):
        ge, le = CMP[r]
        _, zf, zl, _, _ = z_rules[r]
        for u, zsymbol, phase_weight in [
            (first, zf, -10.0),
            (last, zl, 10.0),
        ]:
            p["Wmz"][u, zsymbol] = et
            p["Wm"][u, ge] = 10.0
            p["Wm"][u, le] = 10.0
            p["Wm"][u, PHASE] = phase_weight
            p["bm"][u] = -30.0

    # Selected rule length pack: source_length + 8 * target_length.
    for r, u in enumerate(LENGTH):
        ge, le = CMP[r]
        _, _, _, znl, znr = z_rules[r]
        p["Wmz"][u, znl] = ep
        p["Wmz"][u, znr] = 8.0 * ep
        p["Wm"][u, ge] = 10.0
        p["Wm"][u, le] = 10.0
        p["bm"][u] = -20.0

    for u in LENGTH:
        p["Wm"][RHS_TWO, u] = gain / ep
    p["bm"][RHS_TWO] = gain * (5.0 / ep - 14.0)

    # DONE means this navigation consumes the source group.
    p["Wm"][DONE, PHASE] = 10.0
    p["Wm"][DONE, RHS_TWO] = -10.0
    p["bm"][DONE] = 10.0

    p["Wm"][CONSUME, DONE] = 10.0
    p["Wma"][CONSUME, 4] = 20.0
    p["bm"][CONSUME] = -20.0

    # Analog source-length pulse on consuming navigation.
    for u in LENGTH:
        p["Wm"][ADVANCE, u] = 1.0
    p["Wm"][ADVANCE, RHS_TWO] = -4.0 * ep
    p["Wm"][ADVANCE, DONE] = 10.0
    p["Wma"][ADVANCE, 4] = 20.0
    p["bm"][ADVANCE] = 5.0 - 12.0 * ep - 30.0

    # Set last-target phase after the first target of a two-symbol rule.
    p["Wm"][SET_PHASE, PHASE] = -10.0
    p["Wm"][SET_PHASE, RHS_TWO] = 10.0
    p["Wma"][SET_PHASE, 4] = 20.0
    p["bm"][SET_PHASE] = -30.0

    # Retentive phase latch, rather than delayed self-copy feedback.
    p["Wm"][PHASE, PHASE] = 10.0
    p["Wm"][PHASE, SET_PHASE] = 10.0
    p["Wm"][PHASE, ADVANCE] = -20.0
    p["bm"][PHASE] = -11.0

    # Current-output comparisons against the selected target.
    target_units = [u for pair in TARGET for u in pair]
    for u in target_units:
        p["Wm"][OUT_GE, u] = -gain / et
        p["Wm"][OUT_LE, u] = gain / et
        p["Wm"][FAR_POS, u] = -gain / et
        p["Wm"][FAR_NEG, u] = gain / et

    for u in (OUT_GE, FAR_POS):
        p["Wmz"][u, z_output] = gain
    for u in (OUT_LE, FAR_NEG):
        p["Wmz"][u, z_output] = -gain

    p["bm"][OUT_GE] = gain * (-11.0 / et + 0.25)
    p["bm"][OUT_LE] = gain * (11.0 / et + 0.25)
    p["bm"][FAR_POS] = gain * (-11.0 / et - 3.5)
    p["bm"][FAR_NEG] = gain * (11.0 / et - 3.5)

    p["bm"][NEG_ONE] = -20.0
    p["bm"][INIT] = -10.0
    p["Wma"][INIT, :] = 20.0

    # Extended settling applies only to consuming C-mode navigation.
    p["Wm"][HISTORY[0], DONE] = 10.0
    p["Wma"][HISTORY[0], 4] = 20.0
    p["Wma"][HISTORY[0], 0] = 40.0
    p["bm"][HISTORY[0]] = -20.0
    for previous, current in zip(HISTORY, HISTORY[1:]):
        p["Wm"][current, previous] = 10.0

    p["Wm"][SET_DELAY, SET_PHASE] = 10.0

    p["Wm"][COOL, HISTORY[0]] = 80.0
    for u in HISTORY[1:]:
        p["Wm"][COOL, u] = 10.0
    p["Wm"][COOL, SET_PHASE] = 80.0
    p["Wm"][COOL, SET_DELAY] = 80.0
    p["Wmz"][COOL, z_c_mode] = 60.0
    p["Wma"][COOL, 4] = 160.0
    p["Wma"][COOL, 0] = 160.0
    p["bm"][COOL] = 215.0

    # Alternating settling edits limit net displacement.
    p["Wm"][ALTERNATE, COOL] = 10.0
    p["Wma"][ALTERNATE, 2] = 20.0
    p["bm"][ALTERNATE] = -20.0

    p["bo"][:] = -1000.0

    # Reset only initially in prefix-counted C mode.
    p["bo"][0] = -40.0
    p["Wo"][0, z_c_mode] = 40.0
    p["Wo"][0, 128 + INIT] = -20.0

    direction = {
        OUT_GE: 1.0,
        OUT_LE: -1.0,
        FAR_POS: -2.0,
        FAR_NEG: 2.0,
    }
    for u, w in direction.items():
        p["Wo"][1, 128 + u] = w
        p["Wo"][2, 128 + u] = -w

    p["bo"][1] = 10.0
    p["Wo"][1, 128 + ALTERNATE] = 10.0
    p["bo"][2] = 5.0
    p["Wo"][2, 128 + COOL] = 5.0

    # Navigate only on a settled equality comparison.
    p["bo"][4] = -115.0
    p["Wo"][4, 128 + OUT_GE] = 20.0
    p["Wo"][4, 128 + OUT_LE] = 20.0
    p["Wo"][4, 128 + COOL] = -100.0

    return p
