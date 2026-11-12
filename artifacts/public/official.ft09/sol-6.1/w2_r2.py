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
    p = {name: np.zeros(shape, dtype=np.float32)
         for name, shape in shapes.items()}

    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),             (0, 1),
        (1, -1),  (1, 0),   (1, 1),
    ]

    # Binary coding compatible with the game's visible palette pairs:
    # (9, 8), (8, 12), and (14, 15).
    colour_classes = ((9, 12, 15), (8, 14))
    tile_colours = colour_classes[0] + colour_classes[1]
    clue_marks = (0, 2, 3)

    def add1(out, colours, dy=0, dx=0, weight=1.0):
        for colour in colours:
            p["W1"][out, colour, 2 + dy, 2 + dx] += weight

    def add2(out, channel, dy=0, dx=0, weight=1.0):
        p["W2"][out, channel, 1 + dy, 1 + dx] += weight

    # First-layer channels:
    #   0..7:  clue class 1 AND nonzero directional marker
    #   8..15: clue class 0 AND zero directional marker
    #   16:    clue-centre mask
    #   17,18: ordinary tile centres, separated by colour class
    #   19,20: unique upper-left centre pixel of a special tile
    #   21:    shifted background-colour detector
    #
    # Offsets of two display pixels correspond to one sprite pixel.
    for d, (dy, dx) in enumerate(directions):
        positive = d
        negative = 8 + d

        for out in (positive, negative):
            add1(out, tile_colours, weight=4.0)
            for ey, ex in directions:
                add1(out, clue_marks, 2 * ey, 2 * ex, 4.0)

        add1(positive, colour_classes[1], weight=1.0)
        add1(positive, (2, 3), 2 * dy, 2 * dx, 1.0)
        p["b1"][positive] = -37.0

        add1(negative, colour_classes[1], weight=-1.0)
        add1(negative, (2, 3), 2 * dy, 2 * dx, -1.0)
        p["b1"][negative] = -35.0

    add1(16, tile_colours)
    for dy, dx in directions:
        add1(16, clue_marks, 2 * dy, 2 * dx)
    p["b1"][16] = -8.0

    for bit, colours in enumerate(colour_classes):
        ordinary = 17 + bit
        add1(ordinary, colours)
        for dy, dx in directions:
            add1(ordinary, colours, 2 * dy, 2 * dx)
        p["b1"][ordinary] = -8.0

        special = 19 + bit
        add1(special, colours)
        for dy, dx in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            add1(special, colours, 2 * dy, 2 * dx)
        for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            add1(special, (6,), 2 * dy, 2 * dx)

        # These two transitions select one pixel from the 2x2 centre.
        add1(special, (6,), -1, 0)
        add1(special, (6,), 0, -1)
        p["b1"][special] = -10.0

    add1(21, (4,), -2, -2)

    # Second-layer channels:
    #   0..7:  unique clue seeds requesting colour class 0
    #   8:     unique clue-centre seed, irrespective of requested class
    #   9..24: directional matching thresholds
    #   25:    legal candidate click pixels
    #   26:    exclusion mask for the tutorial's noninteractive examples
    for d in range(8):
        add2(d, d)
        add2(d, 8 + d)
        add2(d, 16, -1, 0, -4.0)
        add2(d, 16, 0, -1, -4.0)
        add2(d, 21, -1, -1)
        p["b2"][d] = -1.0

    add2(8, 16)
    add2(8, 16, -1, 0, -4.0)
    add2(8, 16, 0, -1, -4.0)
    add2(8, 21, -1, -1)
    p["b2"][8] = -1.0

    for d, (dy, dx) in enumerate(directions):
        for bit in range(2):
            out = 9 + 2 * d + bit

            # Ordinary-tile anchor: matching threshold is one.
            add2(out, 17 + bit)
            for previous in (17, 18):
                add2(out, previous, -1, 0, -4.0)
                add2(out, previous, 0, -1, -4.0)

            # Special tiles are tested one step earlier along the ray.
            # The arriving signal is therefore four rather than one.
            add2(out, 19 + bit, dy, dx, 4.0)

    for channel in (17, 18):
        add2(25, channel)
        add2(25, channel, -1, 0, -4.0)
        add2(25, channel, 0, -1, -4.0)

    for dy, dx in directions:
        add2(25, 19, dy, dx)
        add2(25, 20, dy, dx)

    add2(26, 17)
    add2(26, 18)
    add2(26, 21, -1, -1, -1.0)

    # Pass local features through; axial pooling is intentionally unused.
    for channel in range(27):
        p["W3"][channel, channel] = 1.0

    # Sixteen directed streams: two requested colour classes per direction.
    # Eight pixel shifts carry 4**8 to an ordinary neighbour as exactly 1.
    seed_strength = float(4 ** 8)
    for d, (dy, dx) in enumerate(directions):
        zero_stream = 2 * d
        one_stream = zero_stream + 1

        p["Wh"][zero_stream, zero_stream, 1 - dy, 1 - dx] = 0.25
        p["Wh"][one_stream, one_stream, 1 - dy, 1 - dx] = 0.25

        p["Uh"][zero_stream, d] = seed_strength
        p["Uh"][zero_stream, 9 + zero_stream] = -1.0

        # Class-1 seed = clue anchor minus class-0 seed.
        p["Uh"][one_stream, 8] = seed_strength
        p["Uh"][one_stream, d] = -seed_strength
        p["Uh"][one_stream, 9 + one_stream] = -1.0

    # Suppress all locations except candidate tile pixels.
    # The bound exceeds the total possible transported signal.
    mask_strength = float(2 ** 20)
    p["Wc"][0, 25] = mask_strength
    p["Wc"][0, 26] = -mask_strength
    p["Wc"][0, 32:48] = 1.0
    p["bc"][0] = -mask_strength

    # Always use ACTION6 when available.
    p["bo"][6] = 1.0

    return p
