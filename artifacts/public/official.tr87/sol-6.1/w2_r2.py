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

    glyphs = {
        "A": [
            ["..#..", "..#..", ".###.", "..#..", "#####"],
            ["....#", "..#.#", "#####", "#.#..", "#...."],
            ["#####", "#.#..", "#....", "#.#..", "#####"],
            ["..#..", "#####", "#.#.#", "#.#.#", "..#.."],
            ["#...#", "#####", "..#..", "#####", "#...#"],
            ["#####", "..#..", "###..", "#....", "#####"],
            ["#...#", "#####", "#...#", "#...#", "##.##"],
        ],
        "B": [
            ["#....", "####.", "#..#.", "#..#.", "#####"],
            ["#####", "#...#", "###.#", "#.#.#", "#####"],
            ["..###", "..#.#", "#####", "#.#..", "###.."],
            [".###.", ".#.#.", "#####", "#...#", "#####"],
            ["#####", "#...#", "#...#", "#####", "..#.."],
            ["####.", "#..##", "#...#", "##..#", ".####"],
            ["..#..", "#####", "#.#.#", "#####", "..#.."],
        ],
        "C": [
            ["#.#.#", "#....", "#.#.#", "#....", "#####"],
            ["###.#", "#....", ".....", "....#", "#.###"],
            ["#.###", "..#..", "..#..", "..#..", "###.#"],
            ["..#..", "#.#.#", "..#..", "#.#.#", "..#.."],
            ["##.##", "#...#", "#.#.#", "#...#", "##.##"],
            ["#.###", "#.#.#", "###.#", ".....", "#.#.#"],
            ["#...#", ".....", "#...#", "##.##", ".#.#."],
        ],
    }

    # Rotation-invariant foreground counts.
    orbits = []
    seen = {(2, 2)}
    for y in range(5):
        for x in range(5):
            if (y, x) in seen:
                continue
            orbit = []
            yy, xx = y, x
            for _ in range(4):
                orbit.append((yy, xx))
                yy, xx = xx, 4 - yy
            orbit = sorted(set(orbit))
            seen.update(orbit)
            orbits.append(orbit)

    for q, orbit in enumerate(orbits):
        for threshold in range(4):
            channel = 4 * q + threshold
            for y, x in orbit:
                p["W1"][channel, 5, y, x] = 1.0
            p["b1"][channel] = -float(threshold)
    p["W1"][24, 5, 2, 2] = 1.0

    # No glyph has exactly one foreground corner.
    p["W1"][1] = 0.0
    p["b1"][1] = 0.0
    p["W1"][1, 3, 2, 0] = 1.0
    p["W1"][1, 3, 2, 4] = 1.0
    p["W1"][1, 16, 2, 2] = -63.0 / 32.0

    # The inner-corner orbit likewise never has count one.
    inner_hinge = 17
    p["W1"][inner_hinge] = 0.0
    p["W1"][inner_hinge, 16, 2, 2] = 63.0
    p["b1"][inner_hinge] = -32.0

    # Factored exterior-border checks.
    p["W1"][25, 5, 0, :] = 1.0
    p["W1"][25, 5, 4, :] = 1.0
    p["W1"][25, 5, 2, :] = -1.0
    p["b1"][25] = 5.0
    p["W1"][26, 5, :, 4] = 1.0

    for colour in (5, 7, 10):
        p["W1"][27, colour, :, :] = 1.0
    for colour in (5, 7, 11):
        p["W1"][28, colour, :, :] = 1.0

    p["W1"][29, 16, 2, 2] = 63.0
    p["W1"][30, 17, 2, 2] = 63.0
    p["W1"][31, 0, 2, 2] = 1.0

    glyph_penalty = 2048.0
    family_penalty = 8192.0
    position_scale = 0.125
    region_scale = 256.0
    row_scale = 8.0
    baseline_max = 248.0

    def descriptor(rows):
        image = np.array(
            [[character == "#" for character in row] for row in rows],
            dtype=np.int32,
        )
        counts = [
            sum(int(image[y, x]) for y, x in orbit)
            for orbit in orbits
        ]
        return counts, int(image[2, 2])

    def add_detector(out, family, digit, source):
        counts, centre = descriptor(glyphs[family][digit - 1])
        raw_x = 0 if source else 2
        base = 1.0 if source else float(digit)
        p["b2"][out] = base - region_scale

        for q, target in enumerate(counts):
            p["b2"][out] -= glyph_penalty * target
            slope = 1.0 if target == 0 else -1.0
            p["W2"][out, 4 * q, 1, raw_x] -= (
                glyph_penalty * slope
            )
            if 1 <= target <= 3:
                p["W2"][out, 4 * q + target, 1, raw_x] -= (
                    2.0 * glyph_penalty
                )

        p["b2"][out] -= glyph_penalty * centre
        p["W2"][out, 24, 1, raw_x] -= (
            glyph_penalty * (1.0 - 2.0 * centre)
        )

        # These validation scores cannot reward missing foreground.
        if family in ("A", "C"):
            sign = 1.0 if family == "A" else -1.0
            p["W2"][out, 27, 1, raw_x] += sign * family_penalty
            p["W2"][out, 28, 1, raw_x] -= sign * family_penalty
            for q in range(len(orbits)):
                p["W2"][out, 4 * q, 1, raw_x] += family_penalty
            p["W2"][out, 24, 1, raw_x] += family_penalty
            p["b2"][out] -= 25.0 * family_penalty
        else:
            p["W2"][out, 27, 1, raw_x] += family_penalty
            p["W2"][out, 28, 1, raw_x] += family_penalty
            p["b2"][out] -= 50.0 * family_penalty

        p["W2"][out, 25, 0, raw_x] -= glyph_penalty
        p["W2"][out, 25, 2, raw_x] -= glyph_penalty
        p["b2"][out] += 10.0 * glyph_penalty

        if source:
            p["W2"][out, 26, 1, 1] -= glyph_penalty

        connector_x = 2 if source else 0
        p["W2"][out, 1, 1, connector_x] += region_scale
        p["W2"][out, 29, 1, 1] += row_scale
        p["W2"][out, 30, 1, 1] += position_scale

    for j in range(7):
        add_detector(j, "A", j + 1, True)
        add_detector(7 + j, "C", j + 1, True)
        add_detector(14 + j, "A", j + 1, False)
        add_detector(21 + j, "B", j + 1, False)

    # Cursor encoding:
    # top bracket -> 128-row;
    # bottom bracket -> 139-row.
    # The latter aligns the bracket with the input row, eleven pixels
    # above the selected output's centre.
    p["W2"][28, 31, 1, 1] = 128.0
    p["W2"][28, 29, 1, 1] = -1.0
    p["W2"][28, inner_hinge, 1, 1] = 11.0
    p["W2"][28, inner_hinge, 0, 1] = -11.0

    p["W2"][29, 29, 1, 1] = row_scale
    p["W2"][29, 30, 1, 1] = position_scale
    p["b2"][29] = -region_scale
    p["W2"][30, 29, 1, 1] = 1.0

    def local(c):
        return c

    def rowmax(c):
        return 32 + c

    def colmax(c):
        return 64 + c

    # Position comparison for each possible direct-rule input.
    offset = baseline_max - 1.0 + 8.0 * position_scale
    deadband = 0.0625
    for j in range(7):
        positive, negative = 2 * j, 2 * j + 1
        for c in (j, 7 + j):
            p["W3"][positive, rowmax(c)] = 1.0
            p["W3"][negative, rowmax(c)] = -1.0
        p["W3"][positive, colmax(29)] = -1.0
        p["W3"][negative, colmax(29)] = 1.0
        p["b3"][positive] = offset - deadband
        p["b3"][negative] = -offset - deadband

    # Glyph-class evidence at cursor columns. Spatial row filtering
    # happens in H, rather than allowing evidence from other rows to mix.
    for j in range(7):
        out = 14 + j
        for c in (j, 7 + j, 14 + j, 21 + j):
            p["W3"][out, local(c)] = 1000.0
        p["W3"][out, local(29)] = -1000.0
        p["W3"][out, colmax(28)] = 1.0
        p["b3"][out] = -1032.0

    # One-hot current output glyph.
    output_constant = 656.0
    for j in range(7):
        out = 21 + j
        inverse = 1.0 / float(j + 1)
        p["W3"][out, local(14 + j)] = inverse
        p["W3"][out, local(21 + j)] = inverse
        p["W3"][out, colmax(29)] = -inverse
        p["W3"][out, colmax(28)] = row_scale * inverse
        p["b3"][out] = -output_constant * inverse

    # Direct-rule destination number.
    for c in range(14, 28):
        p["W3"][28, local(c)] = 1.0
    p["W3"][28, colmax(29)] = -1.0
    p["W3"][28, local(29)] = -64.0
    p["b3"][28] = baseline_max

    # Top-selector evidence, used to suppress inappropriate direct edits.
    p["W3"][29, colmax(28)] = 1.0
    p["b3"][29] = -100.0

    # Absolute row error relative to the desired observation row.
    p["W3"][30, local(30)] = 1.0
    p["W3"][30, colmax(28)] = 1.0
    p["b3"][30] = -134.0
    p["W3"][31, local(30)] = -1.0
    p["W3"][31, colmax(28)] = -1.0
    p["b3"][31] = 134.0

    # H[0:7] is the complete direct-rule lookup table.
    # H[9:16] identifies the glyph on the cursor-aligned input row.
    # No spatial feedback is required for these computations.
    for j in range(7):
        p["Uh"][j, 28] = 1.0
        p["Uh"][j, 2 * j] = -128.0
        p["Uh"][j, 2 * j + 1] = -128.0

        p["Uh"][9 + j, 14 + j] = 1.0
        p["Uh"][9 + j, 30] = -10000.0
        p["Uh"][9 + j, 31] = -10000.0

    # Select a table entry using a pair of gated hinges. Their
    # difference is the desired digit, not the detector's amplitude.
    desired_coefficients = {}
    z = 0
    for j in range(7):
        hi, lo = z, z + 1
        p["Wz"][hi, j] = 1.0
        p["Wz"][hi, 9 + j] = 1.0
        p["bz"][hi] = -20.0
        p["Wz"][lo, 9 + j] = 1.0
        p["bz"][lo] = -20.0
        desired_coefficients[hi] = 1.0
        desired_coefficients[lo] = -1.0
        z += 2

    current_units = []
    for j in range(7):
        p["Wz"][z, 32 + 21 + j] = 1.0
        current_units.append((z, float(j + 1)))
        z += 1

    mode_unit = z
    p["Wz"][mode_unit, 32 + 29] = 1.0

    # Tanh threshold units implement interval indicators for integer
    # differences. The margins are half-integers.
    thresholds = (-3.5, -0.5, 0.5, 3.5)
    for unit, threshold in enumerate(thresholds):
        for zz, coefficient in desired_coefficients.items():
            p["Wmz"][unit, zz] = 20.0 * coefficient
        for zz, digit in current_units:
            p["Wmz"][unit, zz] = -20.0 * digit
        p["bm"][unit] = -20.0 * threshold

    # Missing direct lookup.
    for zz, coefficient in desired_coefficients.items():
        p["Wmz"][4, zz] = -20.0 * coefficient
    p["bm"][4] = 10.0

    # Top-selector mode.
    p["Wmz"][5, mode_unit] = 4.0
    p["bm"][5] = -10.0

    # Forward: d in {1,2,3,-6,-5,-4}.
    p["Wo"][2, 128 + 0] = -0.5
    p["Wo"][2, 128 + 2] = 0.5
    p["Wo"][2, 128 + 3] = -0.5
    p["bo"][2] = 0.5

    # Reverse: d in {-3,-2,-1,4,5,6}.
    p["Wo"][1, 128 + 0] = 0.5
    p["Wo"][1, 128 + 1] = -0.5
    p["Wo"][1, 128 + 3] = 0.5
    p["bo"][1] = 0.5

    # Correct, missing lookup, or unsupported top-selector operation:
    # advance rather than applying a misleading direct translation.
    p["Wo"][4, 128 + 1] = 0.5
    p["Wo"][4, 128 + 2] = -0.5
    p["Wo"][4, 128 + 4] = 1.0
    p["Wo"][4, 128 + 5] = 1.0
    p["bo"][4] = 2.05

    # After several consecutive advances, make one edit. The direct
    # comparator can undo it and thereby trigger a completion check.
    p["Wma"][8, 4] = 10.0
    p["bm"][8] = -5.0
    for i in range(9, 15):
        p["Wm"][i, i - 1] = 10.0
    p["Wm"][15, 8:15] = 10.0
    p["Wma"][15, 4] = 20.0
    p["bm"][15] = -85.0
    p["Wo"][2, 128 + 15] = 0.6
    p["bo"][2] += 0.6

    p["bo"][3] = -2.0
    for action in (0, 5, 6, 7):
        p["bo"][action] = -100.0

    return p
