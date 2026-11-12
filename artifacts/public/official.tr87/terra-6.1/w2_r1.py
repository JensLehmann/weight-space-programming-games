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

    glyphs_a = [
        ["00100", "00100", "01110", "00100", "11111"],
        ["00001", "00101", "11111", "10100", "10000"],
        ["11111", "10100", "10000", "10100", "11111"],
        ["00100", "11111", "10101", "10101", "00100"],
        ["10001", "11111", "00100", "11111", "10001"],
        ["11111", "00100", "11100", "10000", "11111"],
        ["10001", "11111", "10001", "10001", "11011"],
    ]
    glyphs_b = [
        ["10000", "11110", "10010", "10010", "11111"],
        ["11111", "10001", "11101", "10101", "11111"],
        ["00111", "00101", "11111", "10100", "11100"],
        ["01110", "01010", "11111", "10001", "11111"],
        ["11111", "10001", "10001", "11111", "00100"],
        ["11110", "10011", "10001", "11001", "01111"],
        ["00100", "11111", "10101", "11111", "00100"],
    ]

    # A base-5 signature is invariant under quarter-turn rotations.
    orbit_weights = np.zeros((5, 5), dtype=np.int32)
    visited = set()
    orbit = 0
    for r in range(5):
        for c in range(5):
            if (r, c) in visited:
                continue
            cells = set()
            rr, cc = r, c
            for _ in range(4):
                cells.add((rr, cc))
                rr, cc = cc, 4 - rr
            for rr, cc in cells:
                visited.add((rr, cc))
                orbit_weights[rr, cc] = 5 ** orbit
            orbit += 1

    for k, rows in enumerate(glyphs_a + glyphs_b):
        mask = np.array(
            [[int(ch) for ch in row] for row in rows],
            dtype=np.int32,
        )
        sig = int(np.sum(mask * orbit_weights))
        p["W1"][2 * k, 5] = orbit_weights
        p["b1"][2 * k] = 0.5 - sig
        p["W1"][2 * k + 1, 5] = -orbit_weights
        p["b1"][2 * k + 1] = 0.5 + sig

        p["W2"][k, 2 * k, 1, 1] = -1.0
        p["W2"][k, 2 * k + 1, 1, 1] = -1.0
        border = 28 if k < 7 else 29
        p["W2"][k, border, 2, 2] = 1.0
        p["b2"][k] = -0.5

    for channel, colour in ((28, 10), (29, 7)):
        p["W1"][channel, colour, 4, 2] = 1.0
        p["W1"][channel, colour, 2, 4] = 1.0

    # Five zero pixels detect the horizontal selection bracket.
    p["W1"][30, 0, 2, :] = 1.0
    p["b1"][30] = -4.5
    p["W2"][14, 30, 1, 1] = 2.0

    # Encode row and a small column component in one feature.
    p["W1"][31, 16, 2, 2] = 1.0
    p["W1"][31, 17, 2, 2] = 0.01
    p["W2"][15, 31, 1, 1] = 1.0

    # Raw glyph features and bracket.
    for k in range(14):
        p["W3"][k, k] = 2.0
    p["W3"][14, 14] = 1.0

    # Selected lower glyphs, using the bracket's column maximum.
    for k in range(14):
        out = 15 + k
        p["W3"][out, k] = 2.0
        p["W3"][out, 64 + 14] = 2.0
        p["W3"][out, 15] = 2.0
        p["b3"][out] = -4.05

    # Encoded position of the lowest bracket bar.
    p["W3"][29, 14] = 1.0
    p["W3"][29, 15] = 1.0
    p["b3"][29] = -1.0

    # Positive for a bracket in the upper rule area.
    p["W3"][31, 14] = 1.0
    p["W3"][31, 15] = -2.0
    p["b3"][31] = -0.1

    # Propagate bracket signals vertically. The remaining channels
    # detect a glyph between the two bracket bars.
    p["Wh"][0, 0, 2, 1] = 1.0
    p["Wh"][1, 1, 0, 1] = 1.0
    p["Uh"][0, 14] = 1.0
    p["Uh"][1, 14] = 1.0
    p["bh"][0] = -0.1
    p["bh"][1] = -0.1
    for k in range(14):
        channel = 2 + k
        p["Wh"][channel, 0, 1, 1] = 1.0
        p["Wh"][channel, 1, 1, 1] = 1.0
        p["Uh"][channel, k] = 1.0
        p["bh"][channel] = -1.3

    # Fixed A-to-B heuristic for selected lower glyphs.
    target_b = [0, 1, 5, 2, 4, 5, 6]
    for a in range(7):
        for b in range(7):
            unit = 7 * a + b
            p["Wz"][unit, 32 + 15 + a] = 1.0
            p["Wz"][unit, 32 + 15 + 7 + b] = 0.5
            p["Wz"][unit, 32 + 31] = -5.0
            p["bz"][unit] = -0.5

            forward = (target_b[a] - b) % 7
            backward = (b - target_b[a]) % 7
            if forward == 0:
                action = 4
            elif forward <= backward:
                action = 2
            else:
                action = 1
            p["Wo"][action, unit] = 10.0

    # Two hinges per upper glyph produce a capped detection value.
    for k in range(14):
        z0 = 49 + 2 * k
        p["Wz"][z0, 2 + k] = 10.0
        p["Wz"][z0 + 1, 2 + k] = 10.0
        p["bz"][z0 + 1] = -1.0

    # Position windows for the eight selected groups in the visible
    # upper-rule layout. Values encode 100*row + column.
    positions = [1911, 1921, 1934, 1951, 3118, 3128, 3141, 3151]
    targets = [4, 2, 0, 4, 3, 0, 1, 1]

    for group, position in enumerate(positions):
        base = 77 + 4 * group
        for j, threshold in enumerate(
            (position - 2, position - 1, position + 1, position + 2)
        ):
            p["Wz"][base + j, 32 + 29] = 6300.0
            p["bz"][base + j] = -float(threshold)

    # Memory units act as conjunctions of upper glyph and position window.
    # The tanh outputs are near -1 when inactive and +1 when active.
    for group in range(8):
        for glyph in range(7):
            unit = 7 * group + glyph
            kind = glyph if group % 2 == 0 else 7 + glyph

            z_glyph = 49 + 2 * kind
            p["Wmz"][unit, z_glyph] = 10.0
            p["Wmz"][unit, z_glyph + 1] = -10.0

            z_group = 77 + 4 * group
            for j, value in enumerate((10.0, -10.0, -10.0, 10.0)):
                p["Wmz"][unit, z_group + j] = value
            p["bm"][unit] = -15.0

            desired = targets[group]
            forward = (desired - glyph) % 7
            backward = (glyph - desired) % 7
            if forward == 0:
                action = 4
            elif forward <= backward:
                action = 2
            else:
                action = 1

            p["Wo"][action, 128 + unit] = 5.0
            p["bo"][action] += 5.0

    p["bo"][4] += 0.2
    return p
