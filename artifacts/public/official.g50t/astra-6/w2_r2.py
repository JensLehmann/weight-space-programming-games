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
    p = {name: np.zeros(shape, np.float32)
         for name, shape in shapes.items()}

    W1, b1 = p["W1"], p["b1"]
    W2, b2 = p["W2"], p["b2"]
    W3, b3 = p["W3"], p["b3"]

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    reverse = (2, 1, 4, 3)

    def c1(out, inp, dy=0, dx=0, value=1.0):
        W1[out, inp, dy + 2, dx + 2] += value

    def c2(out, inp, dy=0, dx=0, value=1.0):
        W2[out, inp, dy + 1, dx + 1] += value

    # ------------------------------------------------------------
    # Visual predicates.
    # ------------------------------------------------------------

    # Broad floor rather than thin decorative wiring.
    W1[0, 1:16, :, :] = 1.0
    b1[0] = -24.0

    # Dense gate material, excluding small pressure plates.
    W1[1, 8, :, :] = 1.0
    W1[1, 11, :, :] = 1.0
    b1[1] = -12.0

    # Exact player-center predicate.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            color = 5 if (dy == 0 and dx == 0) else 9
            c1(2, color, dy, dx)
    b1[2] = -24.0

    # Exit center.
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            color = 9 if (dy == 0 and dx == 0) else 5
            c1(3, color, dy, dx)
    b1[3] = -8.0

    # Uncovered pressure plates and toggle switches.
    for out, colors in ((4, (8, 15)), (5, (11,))):
        for color in colors:
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    c1(out, color, dy, dx)
            for dy in (-2, 2):
                for dx in (-2, 2):
                    c1(out, color, dy, dx, -3.0)
        b1[out] = -8.0

    # Thin outward wires remain visible when a plate is occupied.
    # Perpendicular subtraction rejects teleport-frame edges.
    for d, (dy, dx) in enumerate(directions):
        py, px = dx, -dy
        for out, colors in (
            (6 + d, (8, 15)),
            (10 + d, (11,)),
        ):
            for color in colors:
                c1(out, color, 2 * dy, 2 * dx)
                c1(out, color, 2 * dy + py, 2 * dx + px, -0.5)
                c1(out, color, 2 * dy - py, 2 * dx - px, -0.5)

    c1(14, 16)
    c1(15, 17)

    # Active recording icon, and any recording icon.
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            center = dy == 0 and dx == 0
            c1(16, 0 if center else 9, dy, dx)
            colors = (0, 1) if center else (1, 2, 9)
            for color in colors:
                c1(17, color, dy, dx)
    b1[16] = -8.0
    b1[17] = -8.0

    # Teleport-frame corner, detected one pixel above-left of center.
    for dx in range(-2, 3):
        c1(18, 15, -2, dx)
    for dy in range(-1, 3):
        c1(18, 15, dy, -2)
    for dy in (1, 2):
        for dx in (1, 2):
            for color in (2, 5, 9, 14):
                c1(18, color, dy, dx)
    b1[18] = -12.0

    W1[19, 14, :, :] = 1.0
    b1[19] = -12.0

    # ------------------------------------------------------------
    # Second convolution.
    # ------------------------------------------------------------

    c2(0, 0)

    # Expanded gate footprint, subsequently clipped to a binary mask.
    W2[1, 1, :, :] = 1.0
    W2[2, 1, :, :] = 1.0
    b2[2] = -1.0

    for out, inp in ((3, 2), (4, 3), (5, 4), (6, 5)):
        c2(out, inp)

    for out, bias in ((7, -10.0), (8, -11.0)):
        c2(out, 2, value=10.0)
        for d, (dy, dx) in enumerate(directions):
            c2(out, 6 + d, dy, dx)
        b2[out] = bias

    c2(9, 2, value=10.0)
    for d, (dy, dx) in enumerate(directions):
        c2(9, 10 + d, dy, dx)
    b2[9] = -10.0

    # Object-gated coordinates.
    for out, mask, coord in (
        (10, 2, 14), (11, 2, 15),
        (12, 16, 15), (13, 17, 15),
        (22, 3, 14), (23, 3, 15),
    ):
        c2(out, mask)
        c2(out, coord)
        b2[out] = -1.0

    c2(14, 16)
    c2(15, 18, -1, -1)

    c2(16, 18, -1, -1)
    c2(16, 14)
    b2[16] = -1.0

    c2(17, 18, -1, -1)
    c2(17, 14, value=-1.0)

    c2(18, 14)
    c2(19, 15)

    W2[20, 19, :, :] = 1.0
    W2[21, 19, :, :] = 1.0
    b2[21] = -1.0

    # Complementary toggle-column coordinate.
    c2(25, 5)
    c2(25, 15, value=-1.0)

    # ------------------------------------------------------------
    # Axial features.
    # ------------------------------------------------------------

    def local(out, inp, value=1.0):
        W3[out, inp] += value

    def rowmax(out, inp, value=1.0):
        W3[out, 32 + inp] += value

    def colmax(out, inp, value=1.0):
        W3[out, 64 + inp] += value

    local(0, 0)
    local(1, 1)
    local(1, 2, -1.0)
    local(2, 3)
    local(3, 4)
    local(4, 5)

    # Leftmost uncovered toggle on each row.
    local(5, 6, 100.0)
    local(5, 19, -63.0)
    rowmax(5, 25, -63.0)
    b3[5] = -36.0

    local(6, 7)
    local(6, 8, -2.0)

    # Occupied toggle with the same row-wise selection.
    local(7, 9, 100.0)
    local(7, 19, -63.0)
    rowmax(7, 25, -63.0)
    b3[7] = -36.0

    for out, inp in (
        (8, 10), (9, 11),
        (10, 12), (11, 13), (12, 14),
        (13, 18), (14, 19),
        (19, 15), (21, 22), (22, 23),
    ):
        local(out, inp)

    # Provisional vertical links between teleport pads.
    colmax(15, 15)

    colmax(16, 15)
    colmax(16, 17, -1.0)
    local(16, 18, -1.0)
    local(16, 0, -1.0)

    local(17, 18)
    colmax(17, 16, -1.0)
    colmax(17, 15)
    local(17, 0, -1.0)
    b3[17] = -1.0

    local(18, 0)
    colmax(18, 15)
    b3[18] = -1.0

    local(20, 20)
    local(20, 21, -1.0)

    local(24, 3)
    local(24, 15)
    b3[24] = -1.0

    # ------------------------------------------------------------
    # Tied spatial block.
    #
    # Fields:
    #   0: currently traversable exit route
    #   1: currently traversable recording-target route
    #   2: provisional toggle route
    #   3: provisional exit route
    #
    # Each field uses two channels for a clipped flood wave and
    # two for opposing-neighbor arrival-time integrals.
    # ------------------------------------------------------------

    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    for field, source in enumerate((3, 4, 5, 3)):
        a, b, hx, hy = [4 * field + k for k in range(4)]

        for out in (a, b):
            Wh[out, a, :, :] = 1.0
            Wh[out, b, :, :] = -1.0

            Uh[out, 0] = 10.0
            Uh[out, 20] = -10.0
            Uh[out, source] += 1.0
            bh[out] = -10.0

            if field < 2:
                Uh[out, 1] = -10.0
            else:
                Uh[out, 15] += 10.0
                Uh[out, 18] -= 10.0
                Uh[out, 16] -= 2048.0
                Uh[out, 17] -= 2048.0

        bh[b] -= 1.0

        for out in (hx, hy):
            Wh[out, out, 1, 1] = 1.0
            Uh[out, 2] = 1000.0
            bh[out] = -999.0

        Wh[hx, a, 1, 2] = 1.0
        Wh[hx, b, 1, 2] = -1.0
        Wh[hx, a, 1, 0] = -1.0
        Wh[hx, b, 1, 0] = 1.0

        Wh[hy, a, 2, 1] = 1.0
        Wh[hy, b, 2, 1] = -1.0
        Wh[hy, a, 0, 1] = -1.0
        Wh[hy, b, 0, 1] = 1.0

    # ------------------------------------------------------------
    # Pooled feature decoding.
    # ------------------------------------------------------------

    Wz, bz = p["Wz"], p["bz"]

    for field in range(4):
        hx, hy = 4 * field + 2, 4 * field + 3
        components = (
            (hy, -1.0), (hy, 1.0),
            (hx, -1.0), (hx, 1.0),
        )
        for d, (channel, sign) in enumerate(components):
            for offset in (0, 1):
                z = 8 * field + 2 * d + offset
                Wz[z, channel] = sign
                bz[z] = -128.0 * sign - offset

    for z, feature in (
        (32, 8), (33, 9), (34, 6),
        (35, 7), (37, 24),
    ):
        Wz[z, 32 + feature] = 1.0

    Wz[38, 32 + 7] = 1.0
    bz[38] = -1.0

    # Last recording slot is active.
    Wz[36, 32 + 10] = 31.5
    Wz[36, 32 + 11] = -31.5
    bz[36] = 1.0

    for d, (positive, negative) in enumerate(
        ((8, 21), (21, 8), (9, 22), (22, 9))
    ):
        for offset in (0, 1):
            z = 40 + 2 * d + offset
            Wz[z, 32 + positive] = 63.0
            Wz[z, 32 + negative] = -63.0
            bz[z] = -float(offset)

    # ------------------------------------------------------------
    # Memory and policy.
    # ------------------------------------------------------------

    Wm, Wmz = p["Wm"], p["Wmz"]
    Wma, bm = p["Wma"], p["bm"]
    Wo, bo = p["Wo"], p["bo"]

    bo[:] = -10000.0
    bo[1:6] = 0.0

    def gradient(unit, field, direction, weight):
        z = 8 * field + 2 * direction
        Wmz[unit, z] += weight
        Wmz[unit, z + 1] -= weight

    def reach(unit, field, weight):
        for direction in range(4):
            gradient(unit, field, direction, weight)

    def occupied_toggle(unit, weight):
        Wmz[unit, 35] += weight
        Wmz[unit, 38] -= weight

    def sharpen(unit, factor=12.0):
        Wm[unit] *= factor
        Wmz[unit] *= factor
        Wma[unit] *= factor
        bm[unit] *= factor

    def vote(unit, action, weight):
        Wo[action, 128 + unit] += weight
        bo[action] += weight

    # After toggling, pursue the exit rather than revisit the switch.
    Wm[1, 1] = 0.98
    occupied_toggle(1, 5.0)

    coordinate_scale = 0.02
    Wmz[2, 32] = coordinate_scale
    Wmz[3, 33] = coordinate_scale

    for d in range(4):
        action = d + 1

        # Highest movement priority: an ordinary exit route.
        u = 8 + d
        gradient(u, 0, d, 2.0)
        bm[u] = -1.0
        sharpen(u)
        vote(u, action, 12.0)

        # Toggle route when no reachable recording target remains.
        u = 12 + d
        gradient(u, 2, d, 2.0)
        reach(u, 0, -20.0)
        reach(u, 1, -4.0)
        Wm[u, 1] = -4.0
        occupied_toggle(u, -4.0)
        bm[u] = -1.0
        sharpen(u)
        vote(u, action, 10.0)

        # Recording targets take precedence while slots remain.
        u = 16 + d
        gradient(u, 1, d, 2.0)
        reach(u, 0, -20.0)
        Wmz[u, 36] = -4.0
        Wm[u, 1] = -4.0
        occupied_toggle(u, -4.0)
        bm[u] = -1.0
        sharpen(u)
        vote(u, action, 10.0)

        # With all recordings allocated, pursue toggle routes even
        # if an unused ordinary plate remains visible.
        u = 20 + d
        gradient(u, 2, d, 2.0)
        reach(u, 0, -20.0)
        Wmz[u, 36] = 2.0
        Wm[u, 1] = -4.0
        occupied_toggle(u, -4.0)
        bm[u] = -3.0
        sharpen(u)
        vote(u, action, 10.0)

        # Lower-priority provisional exit route.
        u = 24 + d
        gradient(u, 3, d, 2.0)
        reach(u, 0, -20.0)
        bm[u] = -1.0
        sharpen(u)
        vote(u, action, 6.0)

    # Detect and suppress an attempted movement that did not move.
    for d in range(4):
        u = 36 + d
        z = 32 if d < 2 else 33
        old = 2 if d < 2 else 3
        sign = -1.0 if d in (0, 2) else 1.0

        Wmz[u, z] = -sign * 10.5
        Wm[u, old] = sign * 10.5 / coordinate_scale
        Wma[u, d + 1] = 100.0
        bm[u] = -99.5
        sharpen(u, 10.0)
        vote(u, d + 1, -100.0)

        # Small escape tie-break based on recent movement.
        Wma[44 + d, d + 1] = 5.0
        Wm[48 + d, 44 + d] = 5.0
        Wo[reverse[d], 128 + 48 + d] += 0.15

    # Store the last recorded location using paired tanh multiplexers.
    row_store = (4, 5, 6, 7)
    col_store = (40, 41, 42, 43)

    for units, previous_position in (
        (row_store, 2), (col_store, 3)
    ):
        keep_a, keep_b, write_a, write_b = units

        for out in (keep_a, keep_b):
            for inp in units:
                Wm[out, inp] = 0.5

        Wma[keep_a, 5] = 10.0
        Wma[keep_b, 5] = -10.0

        Wm[write_a, previous_position] = 1.0
        Wm[write_b, previous_position] = 1.0
        bm[write_a] = 10.0
        bm[write_b] = -10.0
        Wma[write_a, 5] = -10.0
        Wma[write_b, 5] = 10.0

    # At an unconnected teleport pad, step away and return to advance
    # replaying ghosts until the remote trigger is activated.
    for d in range(4):
        u = 52 + d
        Wmz[u, 37] = 2.0
        Wma[u, d + 1] = 2.0
        reach(u, 0, -20.0)
        bm[u] = -3.0
        sharpen(u)
        vote(u, reverse[d], 100.0)

    # Record at a new pressure-plate location. The larger deadband
    # makes identical-location tests strongly inactive.
    bo[5] = 160.0
    tests = (
        (32, row_store, 1.0),
        (32, row_store, -1.0),
        (33, col_store, 1.0),
        (33, col_store, -1.0),
    )
    for j, (z, units, sign) in enumerate(tests):
        u = 56 + j
        Wmz[u, z] = sign * coordinate_scale * 4000.0
        for inp in units:
            Wm[u, inp] = -sign * 2000.0

        Wmz[u, 34] = 240.0
        Wmz[u, 36] = -400.0
        Wma[u, 5] = -400.0
        bm[u] = -243.0
        Wo[5, 128 + u] = 50.0

    # Weak coordinate fallback beyond the flood horizon.
    for d in range(4):
        u = 60 + d
        z = 40 + 2 * d
        Wmz[u, z] = 2.0
        Wmz[u, z + 1] = -2.0
        for field in range(4):
            reach(u, field, -4.0)
        bm[u] = -1.0
        sharpen(u)
        vote(u, d + 1, 3.0)

    return p
