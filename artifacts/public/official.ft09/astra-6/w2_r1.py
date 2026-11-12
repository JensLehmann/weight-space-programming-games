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

    # Each canonical colour pair straddles this partition:
    # (9, 8), (8, 12), and (14, 15).
    high = (8, 14)
    low = (9, 12, 15)
    palette = high + low
    marks = (0, 2, 3)

    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),            (0, 1),
        (1, -1),  (1, 0),   (1, 1),
    ]

    def add1(out, colours, dy=0, dx=0, weight=1.0):
        for colour in colours:
            p["W1"][out, colour, dy + 2, dx + 2] += weight

    def add2(out, channel, weight=1.0, dy=0, dx=0):
        p["W2"][out, channel, dy + 1, dx + 1] += weight

    # First-layer channels 0..15: the two alternatives requiring
    # a high-class neighbour: high/equal or low/not-equal.
    for d, (dy, dx) in enumerate(directions):
        a, b = 2 * d, 2 * d + 1
        add1(a, high)
        add1(a, (0,), 2 * dy, 2 * dx)
        p["b1"][a] = -1.0

        add1(b, low)
        add1(b, (2, 3), 2 * dy, 2 * dx)
        p["b1"][b] = -1.0

    # Channel 16: a clue centre, anchored uniquely at the upper-left
    # pixel of its central 2x2 block.
    clue = 16
    add1(clue, palette)
    for dy, dx in directions:
        add1(clue, marks, 2 * dy, 2 * dx)
    add1(clue, marks, -1, 0)
    add1(clue, marks, 0, -1)
    p["b1"][clue] = -10.0

    # Channels 17/18: tile interiors of either colour class.
    # Pattern colour 6 is permitted around, but not at, the centre.
    for out, colours in ((17, high), (18, low)):
        add1(out, colours)
        for dy, dx in directions:
            add1(out, colours + (6,), 2 * dy, 2 * dx)
        p["b1"][out] = -8.0

    # Channel 19: ordinary, unpatterned tile interiors.
    for dy in (-2, 0, 2):
        for dx in (-2, 0, 2):
            add1(19, palette, dy, dx)
    p["b1"][19] = -8.0

    # Background probes, extended to offset -3 by the second layer.
    add1(20, (4,), -2, 0)
    add1(21, (4,), 0, -2)

    # Second-layer channels 0..15: uniquely anchored sources,
    # alternating required-high and required-low.
    for d in range(8):
        a, b = 2 * d, 2 * d + 1

        add2(a, a)
        add2(a, b)
        add2(a, clue)
        p["b2"][a] = -1.0

        add2(b, clue)
        add2(b, a, -1.0)
        add2(b, b, -1.0)

    # Unique tile anchors. Requiring navy background above and left
    # also removes the black-backed tutorial demonstrations.
    for out, source in ((16, 17), (17, 18), (18, 19)):
        add2(out, source)
        add2(out, 20, dy=-1)
        add2(out, 21, dx=-1)
        p["b2"][out] = -2.0

    # Axial pooling is unnecessary for this local construction.
    for c in range(19):
        p["W3"][c, c] = 1.0

    # Directional, contracting signal transport.
    # A unit source remains positive at distance eight, but expires
    # before reaching a second neighbouring tile.
    decay = 0.5
    erosion = 0.0015
    ordinary_penalty = 0.00088

    for d, (dy, dx) in enumerate(directions):
        for colour_class in range(2):
            h = 2 * d + colour_class

            p["Wh"][h, h, 1 - dy, 1 - dx] = decay
            p["bh"][h] = -erosion
            p["Uh"][h, h] = 1.0 + erosion

            # Required-high signals vanish on high tiles;
            # required-low signals vanish on low tiles.
            p["Uh"][h, 16 + colour_class] = -2.0

            # Preserve a larger positive margin on patterned tiles,
            # so their collateral effects are handled first.
            p["Uh"][h, 18] = -ordinary_penalty

    # Eligible anchors dominate all off-tile transport signals.
    # Among anchors, only violated constraints contribute positively.
    p["Wc"][0, 16] = 16.0
    p["Wc"][0, 17] = 16.0
    p["Wc"][0, 32:48] = 1.0
    p["bc"][0] = -16.0

    # Always request a click. Memory and global action features
    # are intentionally unused.
    p["bo"][6] = 10.0

    return p
