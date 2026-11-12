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

    # Binary phase encoding for the game's canonical tile palettes:
    # 9/8, 8/12, and 14/15.
    phase = ((8, 14), (9, 12, 15))
    palette = phase[0] + phase[1]
    markers = (0, 2, 3)
    nonzero_markers = (2, 3)
    tile_pixels = palette + (6,)

    # Directions are (dx, dy).
    directions = (
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),            (1, 0),
        (-1, 1),  (0, 1),  (1, 1),
    )
    cardinal = ((-1, 0), (1, 0), (0, -1), (0, 1))
    direction_index = {d: i for i, d in enumerate(directions)}

    def w1_colors(out, colors, dy=0, dx=0, value=1.0):
        for color in colors:
            p["W1"][out, color, 2 + dy, 2 + dx] += value

    def w2(out, inp, dy=0, dx=0, value=1.0):
        p["W2"][out, inp, 1 + dy, 1 + dx] += value

    # W1 channels 0:8:
    #   clue phase + Boolean "ring is nonzero".
    # W1 channels 8:16:
    #   validity of that directed clue observation.
    #
    # The extra checks for diagonal directions ensure that a side-ring
    # entry cannot masquerade as a corner-ring entry.
    for i, (dx, dy) in enumerate(directions):
        w1_colors(i, phase[1], -2 * dy, -2 * dx)
        w1_colors(i, nonzero_markers, -dy, -dx)

        checks = [
            (palette, -2 * dy, -2 * dx),
            (markers, -dy, -dx),
        ]
        if dx != 0 and dy != 0:
            checks.extend([
                (markers, -2 * dy, -dx),
                (markers, -dy, -2 * dx),
            ])

        for colors, oy, ox in checks:
            w1_colors(8 + i, colors, oy, ox)
        p["b1"][8 + i] = -(len(checks) - 1)

    # Tile-centre detectors. Sampling the eight surrounding positions
    # distinguishes six-pixel tiles from clue centres. Marker 6 is allowed
    # so patterned switches receive the same centre detection.
    for bit in range(2):
        out = 16 + bit
        w1_colors(out, phase[bit])
        for dx, dy in directions:
            w1_colors(out, tile_pixels, 2 * dy, 2 * dx)
        p["b1"][out] = -8.0

    # Count switch-mask pixels around a tile centre.
    for dx, dy in directions:
        w1_colors(18, (6,), 2 * dy, 2 * dx)

    # Background probes; W2 extends their offsets from two to three pixels.
    for i, (dx, dy) in enumerate(cardinal):
        w1_colors(19 + i, (4,), 2 * dy, 2 * dx)

    # Clue-centre barrier detector.
    w1_colors(23, palette)
    for dx, dy in directions:
        w1_colors(23, markers, 2 * dy, 2 * dx)
    p["b1"][23] = -8.0

    # Coordinate channels are carried through, but not needed for decisions.
    p["W1"][24, 16, 2, 2] = 1.0
    p["W1"][25, 17, 2, 2] = 1.0

    # At valid directed clue sites, S is 0, 1, or 2.
    # Its parity is the required phase:
    #   required_1 = ReLU(S) - 2 ReLU(S - 1)
    #   required_0 = valid - required_1.
    for i, (dx, dy) in enumerate(directions):
        valid, a, b = 3 * i, 3 * i + 1, 3 * i + 2

        w2(valid, 8 + i, -dy, -dx)

        w2(a, i, -dy, -dx)
        w2(a, 8 + i, -dy, -dx, 8.0)
        p["b2"][a] = -8.0

        w2(b, i, -dy, -dx)
        w2(b, 8 + i, -dy, -dx, 8.0)
        p["b2"][b] = -9.0

        req0, req1 = 2 * i, 2 * i + 1
        p["W3"][req0, valid] = 1.0
        p["W3"][req0, a] = -1.0
        p["W3"][req0, b] = 2.0
        p["W3"][req1, a] = 1.0
        p["W3"][req1, b] = -2.0

    # Interactive tile centres have two background-4 hits among the four
    # probes at distance three. Tutorial examples have background 5 instead.
    for bit in range(2):
        out = 24 + bit
        w2(out, 16 + bit, value=100.0)
        for i, (dx, dy) in enumerate(cardinal):
            w2(out, 19 + i, dy, dx)
        p["b2"][out] = -101.0
        p["W3"][16 + bit, out] = 1.0

    # Binary patterned-switch indicator, allowing any nonempty mask of 6s.
    # The difference of these two rectifiers clips the count to one.
    for out, bias in ((26, -100.0), (27, -101.0)):
        w2(out, 18)
        w2(out, 16, value=100.0)
        w2(out, 17, value=100.0)
        p["b2"][out] = bias
    p["W3"][18, 26] = 1.0
    p["W3"][18, 27] = -1.0

    w2(28, 23)
    p["W3"][19, 28] = 1.0

    w2(29, 24)
    w2(30, 25)
    p["W3"][20, 29] = 1.0
    p["W3"][21, 30] = 1.0

    # Eight recurrent channels: four line orientations, two required phases.
    # Symmetric diffusion plus a negative background drive gives compact
    # requirement waves. Clue centres and matching tile centres are barriers.
    axes = ((1, 0), (0, 1), (1, 1), (1, -1))
    barrier = 10000.0

    for axis, (dx, dy) in enumerate(axes):
        forward = direction_index[(dx, dy)]
        backward = direction_index[(-dx, -dy)]

        for bit in range(2):
            ch = 2 * axis + bit

            p["Wh"][ch, ch, 1 + dy, 1 + dx] = 0.5
            p["Wh"][ch, ch, 1 - dy, 1 - dx] = 0.5

            p["Uh"][ch, 2 * forward + bit] = 16.0
            p["Uh"][ch, 2 * backward + bit] = 16.0
            p["Uh"][ch, 16 + bit] = -barrier
            p["Uh"][ch, 19] = -barrier
            p["bh"][ch] = -1.0

    # A ninth recurrent channel aggregates mismatch evidence only at
    # patterned switches, giving those switches priority.
    p["Wh"][8, :8, 1, 1] = 1.0
    p["Uh"][8, 18] = barrier
    p["bh"][8] = -barrier

    # Restrict clicks to interactive tile centres, then rank by mismatch.
    p["Wc"][0, 16] = 100000.0
    p["Wc"][0, 17] = 100000.0
    p["Wc"][0, 32:40] = 1.0
    p["Wc"][0, 40] = 1000.0

    # Always use ACTION6 when available. Memory and global heads are unused.
    p["bo"][6] = 1.0

    return p
