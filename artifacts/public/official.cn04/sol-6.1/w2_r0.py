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
    p = {name: np.zeros(shape, dtype=np.float32)
         for name, shape in shapes.items()}

    W1, b1 = p["W1"], p["b1"]
    W2, b2 = p["W2"], p["b2"]
    W3 = p["W3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]
    Wz, bz = p["Wz"], p["bz"]
    Wm, Wmz, Wma, bm = (
        p["Wm"], p["Wmz"], p["Wma"], p["bm"]
    )
    Wo, bo, Wc = p["Wo"], p["bo"], p["Wc"]

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    opposite = [1, 0, 3, 2]
    pixel_count_scale = 4096.0 / 3.0

    # Local colour tests exclude the one-pixel status bar.
    # Channels 0..3: red, green, grey, black cell-middle stripes.
    for out, colour in enumerate((8, 3, 4, 0)):
        for dr in (-1, 0, 1):
            W1[out, colour, 2 + dr, 2] = 1.0
        b1[out] = -2.0

    # Two-pixel probes followed by a W2 shift inspect neighbours
    # one rendered cell (three pixels) away.
    for d, (dr, dc) in enumerate(directions):
        rr, cc = 2 + 2 * dr, 2 + 2 * dc
        W1[4 + d, 0, rr, cc] = 1.0
        W1[8 + d, 4, rr, cc] = 1.0
        W1[8 + d, 3, rr, cc] = 1.0

    W1[12, 16, 2, 2] = 1.0
    W1[13, 17, 2, 2] = 1.0
    W1[14, 18, 2, 2] = 1.0

    # Material surrounding connected terminals.
    ordinary_colours = (0, 1, 2, 5, 6, 7, 9, 11, 14)
    for dr, dc in directions:
        rr, cc = 2 + 2 * dr, 2 + 2 * dc
        for colour in ordinary_colours:
            W1[15, colour, rr, cc] = 1.0
        W1[16, 0, rr, cc] = 1.0
        for j, colour in enumerate((10, 12, 15)):
            W1[17 + j, colour, rr, cc] = 1.0

    # Background evidence at the left edge, broadcast axially.
    for j, colour in enumerate((10, 12, 15)):
        W1[20 + j, colour, 2, 2] = 1.0
        W1[20 + j, 17, 2, 2] = -128.0

    for c in range(4):
        W2[c, c, 1, 1] = 1.0

    # Clipped indicator for red terminals adjacent to black.
    for out, bias in ((4, -8.0), (5, -9.0)):
        W2[out, 0, 1, 1] = 8.0
        for d, (dr, dc) in enumerate(directions):
            W2[out, 4 + d, 1 + dr, 1 + dc] = 1.0
        b2[out] = bias

    # Degree-one grey cells. Green neighbours count as occupied:
    # otherwise a connected grey body's last cell becomes a
    # spurious terminal candidate.
    for out, bias in ((6, -6.5), (7, -7.5)):
        W2[out, 2, 1, 1] = 8.0
        for d, (dr, dc) in enumerate(directions):
            W2[out, 8 + d, 1 + dr, 1 + dc] = -1.0
        b2[out] = bias

    W2[8, 12, 1, 1] = 1.0
    W2[9, 13, 1, 1] = 1.0
    W2[10, 14, 1, 1] = 1.0
    W2[11, 2, 1, 1] = 1.0

    # Binary coloured-neighbour tests at green terminals.
    for j, source in enumerate((15, 16, 17, 18, 19)):
        low, high = 12 + 2 * j, 13 + 2 * j
        for out, threshold in ((low, 0.0), (high, 1.0)):
            W2[out, 1, 1, 1] = 256.0
            W2[out, source, :, :] = 1.0
            b2[out] = -256.0 - threshold

    for j in range(3):
        W2[22 + j, 20 + j, 1, 1] = 1.0

    # f3 channels:
    # 0 red; 1 green; 2 inactive grey endpoint;
    # 3 black-adjacent red; 4 black;
    # 5 axial masking evidence; 6 row; 7 column; 8 grey.
    W3[0, 0] = 1.0
    W3[1, 1] = 1.0
    W3[2, 6] = 2.0
    W3[2, 7] = -6.0
    W3[3, 4] = 1.0
    W3[3, 5] = -1.0
    W3[4, 3] = 1.0
    W3[5, 32 + 11] = 1.0
    W3[5, 64 + 11] = 1.0
    W3[6, 8] = 1.0
    W3[7, 9] = 1.0
    W3[8, 2] = 1.0
    W3[9, 10] = 1.0

    for j in range(5):
        W3[10 + j, 12 + 2 * j] = 1.0
        W3[10 + j, 13 + 2 * j] = -1.0
        if j >= 2:
            W3[10 + j, 32 + 22 + j - 2] = -4.0

    # h0-h1: saturated flood broadcasting masking evidence.
    for out in (0, 1):
        for dr, dc in directions:
            Wh[out, 0, 1 + dr, 1 + dc] = 1.0
            Wh[out, 1, 1 + dr, 1 + dc] = -1.0
        Uh[out, 5] = 1.0
    bh[1] = -1.0

    # h2: selected unmatched terminals.
    Uh[2, 0] = 1.0
    Uh[2, 3] = 1.0
    Wh[2, 0, 1, 1] = 1.0
    Wh[2, 1, 1, 1] = -1.0
    bh[2] = -1.0

    # h3: selected connected terminals.
    for c in range(10, 15):
        Uh[3, c] = 1.0
    Wh[3, 0, 1, 1] = 1.0
    Wh[3, 1, 1, 1] = -1.0
    bh[3] = -1.0

    # h4: target terminals, including existing selected connections.
    Uh[4, 0] = 1.0
    Uh[4, 2] = 1.0
    Wh[4, 2, 1, 1] = -1.0
    Wh[4, 3, 1, 1] = 1.0

    # h5: all selected terminals.
    Wh[5, 2, 1, 1] = 1.0
    Wh[5, 3, 1, 1] = 1.0

    # h6-h9: selected/target coordinate sums and maxima.
    for out, mask, coordinate in (
        (6, 5, 6), (7, 5, 7),
        (8, 4, 6), (9, 4, 7),
    ):
        Uh[out, coordinate] = 1.0
        Wh[out, mask, 1, 1] = 1.0
        bh[out] = -1.0

    # h10-h11: narrow and broad attraction fields.
    for dr, dc in directions:
        Wh[10, 10, 1 + dr, 1 + dc] = 0.22
        Wh[11, 11, 1 + dr, 1 + dc] = 0.2495
    Wh[10, 10, 1, 1] = 0.10
    Wh[10, 4, 1, 1] = 1.0
    Wh[11, 4, 1, 1] = 1.0

    # h12-h15: directional field samples at selected terminals.
    for d, (dr, dc) in enumerate(directions):
        out = 12 + d
        Wh[out, 10, 1 + dr, 1 + dc] = 0.8
        Wh[out, 11, 1 + dr, 1 + dc] = 0.2
        Wh[out, 5, 1, 1] = 128.0
        bh[out] = -128.0

    selected_count = np.zeros(96, dtype=np.float32)
    target_count = np.zeros(96, dtype=np.float32)
    red_count = np.zeros(96, dtype=np.float32)
    green_count = np.zeros(96, dtype=np.float32)

    selected_count[16 + 5] = pixel_count_scale
    target_count[16 + 4] = pixel_count_scale
    red_count[16 + 2] = pixel_count_scale
    green_count[64 + 1] = pixel_count_scale

    for d in range(4):
        Wz[d, 16 + 12 + d] = pixel_count_scale
        Wz[d, 16 + 12 + opposite[d]] = -pixel_count_scale

    Wz[4] = -selected_count
    bz[4] = 0.5

    # Visible extension marker in a masked observation.
    Wz[5, 32 + 4] = 1.0
    Wz[5, 0] = 1.0
    Wz[5, 1] = -1.0
    bz[5] = -1.5

    Wz[6] = green_count
    Wz[7] = -red_count
    bz[7] = 0.5

    Wz[8] = selected_count
    bz[8] = -3.5

    Wz[9] = selected_count - target_count
    bz[9] = 1.0

    # Two-terminal half-span discrepancies.
    for axis, (sc, tc) in enumerate(((6, 8), (7, 9))):
        v = np.zeros(96, dtype=np.float32)
        v[tc] = 1.0
        v[sc] = -1.0
        v[16 + tc] = -0.5 * pixel_count_scale
        v[16 + sc] = 0.5 * pixel_count_scale
        Wz[10 + 2 * axis] = v
        Wz[11 + 2 * axis] = -v
        bz[10 + 2 * axis] = -0.010
        bz[11 + 2 * axis] = -0.010

    # Count-conditioned centroid normalization.
    normalization_terms = []
    for group, count, row_channel, col_channel in (
        (0, selected_count, 6, 7),
        (1, target_count, 8, 9),
    ):
        for n in range(1, 13):
            base = 16 + group * 36 + (n - 1) * 3
            gate = 32.0 * count
            bias = 32.0 * (0.5 - n)

            Wz[base] = gate
            bz[base] = bias

            for axis, channel in enumerate((row_channel, col_channel)):
                Wz[base + 1 + axis] = gate
                Wz[base + 1 + axis, 16 + channel] += pixel_count_scale
                bz[base + 1 + axis] = bias

            coefficient = (
                1.0 if n == 1
                else 1.0 / n - 1.0 / (n - 1)
            )
            normalization_terms.append((group, base, coefficient))

    Wz[88] = selected_count
    bz[88] = -2.0
    Wz[89] = -selected_count
    bz[89] = 2.0
    Wz[90] = target_count
    bz[90] = -2.0
    Wz[91] = -target_count
    bz[91] = 2.0
    Wz[94] = red_count

    # Memory 0-7: previous-action indicators.
    for action in range(8):
        Wma[action, action] = 10.0

    # Memory 8: previous displayed connection count.
    Wmz[8, 6] = 0.01

    # Memory 9-10: repeated-rotation counter and switch trigger.
    Wm[9, 9] = 0.98
    Wma[9, 5] = 0.22
    Wma[9, 6] = -0.65
    bm[9] = 0.005

    Wm[10, 9] = 20.0 * 0.98
    Wma[10, 5] = 20.0 * 2.22
    Wma[10, 6] = -13.0
    Wmz[10, 5] = -1000.0
    bm[10] = 20.0 * (0.005 - 0.68 - 2.0)

    # Memory 16-19: immediate movement reversals.
    for d in range(4):
        Wm[16 + d, 1 + opposite[d]] = 20.0
        Wma[16 + d, 1 + d] = 20.0
        bm[16 + d] = -30.0

    Wm[15, 15] = 0.80
    Wm[15, 16:20] = 0.075
    bm[15] = 0.30
    Wma[15, 5] = -1.0
    Wma[15, 6] = -1.0

    Wm[20, 15] = 20.0
    bm[20] = -9.0

    # Span comparison is enabled only for two-terminal sets.
    Wmz[21, 10:14] = 200.0
    Wmz[21, 88:92] = -100.0
    bm[21] = -5.0

    # Switch after gaining a connection if other loose pieces remain.
    Wmz[22, 6] = 10.0
    Wm[22, 8] = -1000.0
    Wmz[22, 9] = -100.0
    bm[22] = -1.0

    # Switch an already-completed piece in a larger assembly.
    Wmz[23, 94] = -20.0
    Wmz[23, 9] = -100.0
    bm[23] = 9.0

    # Movement: attraction gradients plus centroid displacement.
    for d in range(4):
        action = 1 + d
        Wo[action, d] = 50.0
        Wo[action, opposite[d]] = -50.0
        Wo[action, 128 + 1 + opposite[d]] = -0.10
        bo[action] = -0.03

    for group, base, coefficient in normalization_terms:
        sign = -1.0 if group == 0 else 1.0
        for axis, positive_action, negative_action in (
            (0, 2, 1), (1, 4, 3)
        ):
            weight = 100.0 * sign * coefficient
            coordinate_unit = base + 1 + axis
            Wo[positive_action, coordinate_unit] += weight
            Wo[positive_action, base] -= weight
            Wo[negative_action, coordinate_unit] -= weight
            Wo[negative_action, base] += weight

    # Normal rotation baseline is approximately zero.
    # Span mismatch contributes at most 20, rather than 10000.
    bo[5] = 20.002
    Wo[5, 128 + 21] = 10.0
    Wo[5, 128 + 20] = 10.0
    Wo[5, 5] = 10000.0
    Wo[5, 7] = 50.0

    # Switch after progress, completion, or several failed rotations.
    # Final two-piece alignment receives an additional grace period.
    bo[6] = 149.80
    Wo[6, 128 + 10] = 50.0
    Wo[6, 128 + 22] = 50.0
    Wo[6, 128 + 23] = 50.0
    Wo[6, 9] = -25.0
    Wo[6, 4] = 1000.0
    Wo[6, 8] = 30.0

    bo[0] = -1e6
    bo[7] = -1e6

    # Click inactive unmatched terminals, excluding selected greens.
    Wc[0, 32 + 4] = 100.0
    Wc[0, 32 + 3] = -100.0
    Wc[0, 8] = 0.15
    Wc[0, 1] = -1.0
    Wc[0, 6] = 0.025
    Wc[0, 7] = -0.0001

    return p
