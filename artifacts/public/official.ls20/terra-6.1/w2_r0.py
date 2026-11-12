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

    # Coordinate hinges for the status icon.
    for j, col in enumerate((3, 5, 7)):
        W1[2 * j, 17, 2, 2] = 63
        b1[2 * j] = -col - 0.1
        W1[2 * j + 1, 17, 2, 2] = -63
        b1[2 * j + 1] = col - 0.1

    for j, row in enumerate((55, 57, 59)):
        k = 6 + 2 * j
        W1[k, 16, 2, 2] = 63
        b1[k] = -row - 0.1
        W1[k + 1, 16, 2, 2] = -63
        b1[k + 1] = row - 0.1

    # Glyph occupancy and colour bits.
    for color in (12, 9, 14, 8):
        W1[12, color, 2, 2] = 1
    for color in (9, 8):
        W1[13, color, 2, 2] = 1
    for color in (14, 8):
        W1[14, color, 2, 2] = 1

    # Black perimeter surrounding a goal glyph.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if abs(dx) == 2 or abs(dy) == 2:
                W1[15, 5, dy + 2, dx + 2] = 2
    b1[15] = -31

    def pattern_filter(channel, pattern):
        for y in range(3):
            for x in range(3):
                W1[channel, pattern[y][x], y + 1, x + 1] = 2
        b1[channel] = -17

    pattern_filter(16, [[3, 0, 3], [1, 0, 0], [3, 1, 3]])
    pattern_filter(17, [[0, 3, 3], [3, 0, 0], [3, 0, 3]])
    pattern_filter(18, [[9, 14, 14], [9, 0, 8], [12, 12, 8]])

    # Player body.
    for y in range(5):
        for x in range(5):
            color = 12 if y < 2 else 9
            W1[19, color, y, x] = 2
    b1[19] = -49

    W1[20, 4, 2, 2] = 1

    # Suppress the status display when looking for goal glyphs.
    W1[21, 16, 2, 2] = 63
    b1[21] = -52.1

    for i, (dy, dx) in enumerate(directions):
        W1[22 + i, 4, 2 + 2 * dy, 2 + 2 * dx] = 1

    # 1 - row / 63.
    W1[26, 16, 2, 2] = -1
    b1[26] = 1

    # Status glyph bits.
    for r in range(3):
        for c in range(3):
            out = 3 * r + c
            W2[out, 12, 1, 1] = 1
            for channel in (2 * c, 2 * c + 1,
                            6 + 2 * r, 7 + 2 * r):
                W2[out, channel, 1, 1] = -2

    # Goal glyph bits.
    for r in range(3):
        for c in range(3):
            out = 9 + 3 * r + c
            W2[out, 15, 1, 1] = 1
            W2[out, 12, r, c] = 1
            W2[out, 21, 1, 1] = -4
            b2[out] = -1

    # Status colour bits.
    for bit in range(2):
        out = 18 + bit
        W2[out, 13 + bit, 1, 1] = 1
        for channel in (1, 4, 7, 10):
            W2[out, channel, 1, 1] = -2

    # Goal colour bits.
    for bit in range(2):
        out = 20 + bit
        W2[out, 15, 1, 1] = 1
        W2[out, 21, 1, 1] = -4
        b2[out] = -1
        for r in range(3):
            for c in range(3):
                W2[out, 13 + bit, r, c] = 0.1

    # Station values encode q = 0.001 * col - 0.00001 * row.
    # Channel 0 is approximately col - 3.1 in the playable area.
    for out, source in ((22, 16), (23, 17), (24, 18)):
        W2[out, source, 1, 1] = 1
        W2[out, 0, 1, 1] = 0.001
        W2[out, 26, 1, 1] = 0.00063
        b2[out] = -1 + 0.0031 - 0.00063

    # Goal priority also carries q. Its glyph-count term is removed
    # later when decoding coordinates.
    W2[25, 15, 1, 1] = 1
    W2[25, 21, 1, 1] = -4
    W2[25, 0, 1, 1] = 0.001
    W2[25, 26, 1, 1] = 0.00063
    b2[25] = -1.25 + 0.0031 - 0.00063
    for r in range(3):
        for c in range(3):
            W2[25, 12, r, c] = 0.1

    W2[26, 19, 1, 1] = 1
    W2[27, 20, 1, 1] = 1

    # Directional wall flags. The first two also carry player x and y.
    for i, (dy, dx) in enumerate(directions):
        out = 28 + i
        W2[out, 19, 1, 1] = 2
        W2[out, 22 + i, 1 + dy, 1 + dx] = 1
        b2[out] = -2

    W2[28, 0, 1, 1] = 0.001
    b2[28] += 0.0031
    W2[29, 26, 1, 1] = -0.063
    b2[29] += 0.063

    for i in range(32):
        W3[i, i] = 1

    # Station fields receive roughly unit-strength sources.
    for i in (22, 23, 24):
        W3[i, i] = 20

    # Prefer one goal when multiple goals share a column.
    for i in list(range(9, 18)) + [20, 21, 25]:
        W3[i, 25] += 1000
        W3[i, 64 + 25] -= 1000

    # Four diffusion fields: rotation, shape, colour, goal.
    for field, source in enumerate((22, 23, 24, 25)):
        for dy, dx in directions:
            Wh[field, field, 1 + dy, 1 + dx] = 0.2475
        Uh[field, source] = 1
        Uh[field, 27] = -100

    # Sample each field's local directional differences at the player.
    for field in range(4):
        xch = 4 + 3 * field
        ych = xch + 1
        mid = xch + 2

        Wh[xch, field, 1, 2] = 0.5
        Wh[xch, field, 0, 1] = 0.25
        Wh[xch, field, 2, 1] = 0.25

        Wh[ych, field, 2, 1] = 0.5
        Wh[ych, field, 1, 0] = 0.25
        Wh[ych, field, 1, 2] = 0.25

        for dy, dx in directions:
            Wh[mid, field, 1 + dy, 1 + dx] = 0.25

        for ch in (xch, ych, mid):
            Uh[ch, 26] = 100
            bh[ch] = -100

    fmax = 32

    # Shape classifiers use rotation-invariant counts:
    # centre, edge-middle pixels, and corner pixels.
    shape_scores = [
        (3, 0, 2, 9),
        (10, -3, -1, 2),
        (-4, 2, 2, 14),
        (-4, 2, -1, 7),
        (6, 2, -1, 11),
        (-4, -2, 3, 8),
    ]
    centre = [4]
    edges = [1, 3, 5, 7]
    corners = [0, 2, 6, 8]

    for unit, (wc, we, wk, best) in enumerate(shape_scores):
        for offset in (0, 9):
            for j in centre:
                Wz[unit, fmax + offset + j] = wc
            for j in edges:
                Wz[unit, fmax + offset + j] = we
            for j in corners:
                Wz[unit, fmax + offset + j] = wk
        bz[unit] = 1 - 2 * best

    # Colour discrepancies.
    for bit in range(2):
        ui = fmax + 18 + bit
        goal = fmax + 20 + bit
        a = 6 + 2 * bit
        Wz[a, goal] = 1
        Wz[a, ui] = -1
        Wz[a + 1, ui] = 1
        Wz[a + 1, goal] = -4

    # Oriented glyph discrepancy.
    for j in range(9):
        weight = float(1 << j)
        Wz[10, fmax + j] = weight
        Wz[10, fmax + 9 + j] = -weight
        Wz[11, fmax + j] = -weight
        Wz[11, fmax + 9 + j] = weight
    bz[10] = -0.25
    bz[11] = -0.25

    # Local field samples.
    for i in range(12):
        Wz[12 + i, 4 + i] = 1

    # Wall indicators.
    for i in range(4):
        Wz[24 + i, fmax + 28 + i] = 1
        bz[24 + i] = -0.5

    # Recover player coordinate despite the possible added wall flag.
    # For v = q or 1 + q:
    # q = v - 2 relu(v - 0.5) + 2 relu(v - 1).
    for axis in range(2):
        base = 28 + 3 * axis
        source = fmax + 28 + axis
        Wz[base, source] = 1
        Wz[base + 1, source] = 1
        bz[base + 1] = -0.5
        Wz[base + 2, source] = 1
        bz[base + 2] = -1

    # Linear form for each encoded source coordinate.
    q_forms = []
    for field in range(4):
        coeff = np.zeros(96, dtype=np.float32)
        bias = 0.0
        if field < 3:
            coeff[fmax + 22 + field] = 0.05
        else:
            coeff[fmax + 25] = 1
            for j in range(9):
                coeff[fmax + 9 + j] = -0.1
            bias = 0.25
        q_forms.append((coeff, bias))

    # Four q values and ten clipped staircase steps per source.
    for field, (coeff, bias) in enumerate(q_forms):
        Wz[34 + field] = coeff
        bz[34 + field] = bias

        for k in range(10):
            threshold = 0.008 + 0.005 * k
            unit = 38 + 20 * field + 2 * k
            Wz[unit] = coeff
            bz[unit] = bias - threshold
            Wz[unit + 1] = coeff
            bz[unit + 1] = bias - threshold - 0.001

    gates = {"s": 0, "b": 1, "c": 2, "d": 3}

    def set_gate(unit, name):
        if name == "s":
            Wmz[unit, :6] = -12
        else:
            Wmz[unit, :6] = 12
            bm[unit] = -12
            if name in ("c", "d"):
                Wmz[unit, 6:10] = -40
            if name == "d":
                Wmz[unit, 10:12] = -12

    for name, unit in gates.items():
        set_gate(unit, name)

    def add_player_coordinate(unit, axis, scale):
        base = 28 + 3 * axis
        Wmz[unit, base] += scale * 1000
        Wmz[unit, base + 1] -= scale * 2000
        Wmz[unit, base + 2] += scale * 2000

    def add_source_coordinate(unit, field, axis, scale):
        # Target x = 6 + 5000 * sum(clipped staircase steps).
        # Target y = 100000 * (0.001 * x - q).
        if axis == 0:
            bm[unit] += scale * 6
            factor = scale * 5000
        else:
            bm[unit] += scale * 600
            factor = scale * 500000
            Wmz[unit, 34 + field] -= scale * 100000

        for k in range(10):
            z = 38 + 20 * field + 2 * k
            Wmz[unit, z] += factor
            Wmz[unit, z + 1] -= factor

    # A gated directional signal is tanh(gate + signal) - tanh(gate).
    terms = [
        (1, "s", 1),
        (2, "b", 1),
        (2, "c", -1),
        (0, "c", 1),
        (0, "d", -1),
        (3, "d", 1),
    ]

    next_mem = 4
    for field, gate_name, sign in terms:
        baseline = gates[gate_name]
        sample = 12 + 3 * field

        for axis in range(2):
            unit = next_mem
            next_mem += 1
            set_gate(unit, gate_name)

            # Local diffusion gradient.
            Wmz[unit, sample + axis] += 64
            Wmz[unit, sample + 2] -= 64

            # Long-range coordinate guidance.
            add_source_coordinate(unit, field, axis, 0.01)
            add_player_coordinate(unit, axis, -0.01)

            if axis == 0:
                positive_action, negative_action = 4, 3
            else:
                positive_action, negative_action = 2, 1

            Wo[positive_action, 128 + unit] += sign
            Wo[positive_action, 128 + baseline] -= sign
            Wo[negative_action, 128 + unit] -= sign
            Wo[negative_action, 128 + baseline] += sign

    # Mask visibly blocked actions.
    for i in range(4):
        unit = next_mem
        next_mem += 1
        Wmz[unit, 24 + i] = 10
        Wo[1 + i, 128 + unit] = -10

    # Discourage immediately reversing the previous action.
    opposite = {1: 2, 2: 1, 3: 4, 4: 3}
    for action in range(1, 5):
        unit = next_mem
        next_mem += 1
        Wma[unit, action] = 5
        Wo[opposite[action], 128 + unit] = -1.5

    for action in (0, 5, 6, 7):
        bo[action] = -10000

    return p
