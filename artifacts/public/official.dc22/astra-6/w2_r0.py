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
    p = {name: np.zeros(shape, np.float32)
         for name, shape in shapes.items()}

    def c1(o, i, y=0, x=0, v=1.0):
        p["W1"][o, i, y + 2, x + 2] += v

    def c2(o, i, y=0, x=0, v=1.0):
        p["W2"][o, i, y + 1, x + 1] += v

    def lin(o, i, v=1.0, axis=None):
        offset = {None: 0, "row": 32, "col": 64}[axis]
        p["W3"][o, i + offset] += v

    def rec(o, i, y=0, x=0, v=1.0):
        p["Wh"][o, i, y + 1, x + 1] += v

    square = ((0, 0), (0, 1), (1, 0), (1, 1))
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    walk_colours = (1, 2, 6, 7, 8, 9, 11, 12, 13, 14, 15)

    # First layer: actual support and individual supported pixels.
    for colour in walk_colours:
        c1(22, colour)
        for y, x in square:
            c1(0, colour, y, x)
    p["b1"][0] = -3.0

    # Missing checkerboard pixels, clipped to [0,1] by paired ReLUs.
    # Player pixels can stand in for an occluded coloured neighbour.
    for colour, lo, hi in ((9, 1, 2), (15, 17, 18)):
        for out in (lo, hi):
            c1(out, 4, v=5.0)
            for y, x in directions:
                c1(out, colour, y, x)
                c1(out, 14, y, x)
        p["b1"][lo] = -6.0
        p["b1"][hi] = -7.0

    # Unique top-left player pixel.
    c1(3, 14)
    c1(3, 14, -1, 0, -1.0)
    c1(3, 14, 0, -1, -1.0)

    # Isolated 2x2 goal, excluding the larger same-colour control.
    for y, x in square:
        c1(4, 11, y, x)
    for y, x in (
        (-1, 0), (-1, 1), (2, 0), (2, 1),
        (0, -1), (1, -1), (0, 2), (1, 2),
    ):
        c1(4, 11, y, x, -1.0)
    p["b1"][4] = -3.0

    # Portal texture.
    for y, x in square:
        c1(5, 6, y, x)
        c1(5, 7, y, x)
    p["b1"][5] = -3.0

    c1(6, 13)
    c1(11, 16)
    c1(12, 17)

    # Main controls have five-pixel vertical interiors.
    for out, colour in zip(range(13, 17), (8, 9, 6, 15)):
        for y in range(-2, 3):
            c1(out, colour, y, 0)
        c1(out, 17)
        p["b1"][out] = -5.64

    # Local player density.
    for y in range(-2, 3):
        for x in range(-2, 3):
            c1(19, 14, y, x)

    # Pink collectible on a red pad.
    for y, x in square:
        c1(20, 15, y, x)
    c1(20, 2, 0, -1)
    c1(20, 2, 0, 2)
    p["b1"][20] = -5.0

    # Broad pivot neighbourhood for switch-selection cues.
    for y in range(-2, 3):
        for x in range(-2, 3):
            c1(23, 13, y, x)
            c1(24, 13, y, x)
    p["b1"][24] = -1.0

    # Pivot intersection in footprint coordinates: a top-left position
    # is included whenever its 2x2 footprint intersects the pivot.
    for y, x in square:
        c1(25, 13, y, x)
        c1(26, 13, y, x)
    p["b1"][26] = -1.0

    # Second layer: actual and optimistic terrain.
    c2(0, 0)
    for out in (1, 30):
        for y, x in square:
            c2(out, 22, y, x)
            c2(out, 1, y, x)
            c2(out, 2, y, x, -1.0)
            if out == 30:
                c2(out, 17, y, x)
                c2(out, 18, y, x, -1.0)
        p["b2"][out] = -3.0

    for out, inp in (
        (2, 3), (3, 4), (4, 5),
        (10, 11), (11, 12),
        (12, 13), (13, 14), (14, 15), (15, 16),
        (29, 20),
    ):
        c2(out, inp)

    # Binary footprint-pivot mask.
    c2(5, 25)
    c2(5, 26, v=-1.0)

    # Coordinate-valued versions of that same expanded mask.
    for out, coordinate, sign in (
        (6, 11, 1.0), (7, 11, -1.0),
        (8, 12, 1.0), (9, 12, -1.0),
    ):
        c2(out, 25)
        c2(out, 26, v=-1.0)
        c2(out, coordinate, v=sign)
        p["b2"][out] = -1.0 if sign > 0 else 0.0

    c2(16, 23)
    c2(16, 24, v=-1.0)

    # Local density, intersected with genuinely missing checker pixels.
    for out in (17, 18, 19):
        for y in range(-1, 2):
            for x in range(-1, 2):
                c2(out, 19, y, x, 1.0 / 9.0)
    p["b2"][17] = -4.0
    p["b2"][19] = -4.0
    c2(17, 1, v=4.0)
    c2(17, 2, v=-4.0)
    c2(19, 17, v=4.0)
    c2(19, 18, v=-4.0)

    # Directional midpoint masks and currently supported destinations.
    for d, (y, x) in enumerate(directions):
        c2(20 + d, 3, -y, -x)
        c2(24 + d, 3, -y, -x)
        c2(24 + d, 0, y, x)
        p["b2"][24 + d] = -1.0
    p["b2"][28] = 1.0

    # Third layer: terrain, landmarks, and action-aligned masks.
    for out, inp in (
        (1, 1), (2, 3), (3, 4), (4, 2),
        (29, 29), (30, 30),
    ):
        lin(out, inp)
    for d in range(4):
        lin(5 + d, 20 + d)
        lin(9 + d, 24 + d)
    for out, inp in zip(range(13, 17), range(12, 16)):
        lin(out, inp)

    # Exclude everything except main-control interiors from clicks.
    p["b3"][17] = 1.0
    for inp in range(12, 16):
        lin(17, inp, -100.0)

    # Two blue cues: horizontal and vertical approaches to a pivot.
    # Keeping them separate implements a cross rather than a small box.
    lin(0, 18)
    lin(0, 16, 4.0, "row")
    p["b3"][0] = -4.0

    lin(18, 18)
    lin(18, 16, 4.0, "col")
    p["b3"][18] = -4.0

    lin(19, 17, axis="row")
    lin(20, 18, axis="row")
    lin(21, 19, axis="row")

    # Distance penalties outside the prospective bridge arms.
    # Both the axes and their coordinate bounds use footprint pivots.
    lin(22, 5, 63.0, "col")
    lin(22, 7, -63.0, "col")
    lin(22, 10, -63.0)
    p["b3"][22] = -8.0

    lin(23, 10, 63.0)
    lin(23, 6, -63.0, "col")
    lin(23, 5, 63.0, "col")
    p["b3"][23] = -71.0

    lin(24, 5, 63.0, "row")
    lin(24, 9, -63.0, "row")
    lin(24, 11, -63.0)
    p["b3"][24] = -8.0

    lin(25, 11, 63.0)
    lin(25, 8, -63.0, "row")
    lin(25, 5, 63.0, "row")
    p["b3"][25] = -71.0

    lin(26, 5, axis="col")
    lin(27, 5, axis="row")
    lin(28, 28)

    # Visible pink-control presence, broadcast across its row.
    lin(31, 15, 100.0, "row")

    # Recurrent channel 0: prospective bridge support.
    p["Uh"][0, 26:28] = 1.0
    p["Uh"][0, 22:26] = -1.0

    # Channels 1 minus 15: clipped availability wave for pink controls.
    for out in (1, 15):
        for y in (-1, 0, 1):
            rec(out, 1, y, 0)
            rec(out, 15, y, 0, -1.0)
        p["Uh"][out, 31] = 1.0
    p["bh"][15] = -1.0

    # Channel 2: blocked cells. Pink-only terrain needs availability.
    p["bh"][2] = 2.0
    p["Uh"][2, 1] = -2.0
    p["Uh"][2, 30] = -1.0
    rec(2, 0, v=-2.0)
    rec(2, 1, v=-1.0)
    rec(2, 15)

    # Channels 3 minus 4: clipped reachability.
    # Goals and collectibles seed immediately; portals seed weakly.
    for out in (3, 4):
        for y, x in directions:
            rec(out, 3, y, x)
            rec(out, 4, y, x, -1.0)
        rec(out, 2, v=-8.0)
        p["Uh"][out, 2] = 1.0
        p["Uh"][out, 29] = 1.0
        p["Uh"][out, 3] = 1.0e-9
    p["bh"][4] = -1.0

    # Channel 5 integrates arrival time into a spatial potential.
    rec(5, 5)
    rec(5, 3)
    rec(5, 4, v=-1.0)

    # Channels 6..10 sample the current and four destination potentials.
    rec(6, 5)
    p["Uh"][6, 4] = 256.0
    p["bh"][6] = -256.0
    for d, (y, x) in enumerate(directions):
        out = 7 + d
        rec(out, 5, y, x)
        p["Uh"][out, 5 + d] = 256.0
        p["bh"][out] = -256.0

    # Channels 11..14 carry colour-specific control-selection cues.
    for kind in range(4):
        out = 11 + kind
        for y, x in directions:
            rec(out, out, y, x, 0.25)

        strength = (0.35, 1.5, 1.0e-7, 1.5)[kind]
        p["Uh"][out, 18 + kind] = strength
        if kind == 0:
            p["Uh"][out, 0] = strength

        # A field may enter its own control, but not other controls.
        for other in range(4):
            if other != kind:
                p["Uh"][out, 13 + other] = -100000.0

    # Readout of sampled potentials and supported-move masks.
    for d in range(4):
        p["Wz"][d, 7 + d] = 1.0
        p["Wz"][5 + d, 32 + 9 + d] = 1.0
    p["Wz"][4, 6] = 1.0

    # Small reversal penalties, plus a one-step click trace.
    for d, opposite in enumerate((2, 1, 4, 3)):
        p["Wma"][d, opposite] = 2.0
    p["Wma"][4, 6] = 2.0

    p["bo"][:] = -1000.0
    for d in range(4):
        action = 1 + d
        p["Wo"][action, d] = 1.0
        p["Wo"][action, 5 + d] = 200.0
        p["Wo"][action, 128 + d] = -0.015
        p["bo"][action] = -200.0

    # Click when supported movement does not improve the potential.
    p["Wo"][6, 4] = 1.0
    p["Wo"][6, 132] = -0.065
    p["bo"][6] = 0.035

    p["Wc"][0, 17] = -10000.0
    for channel in range(11, 15):
        p["Wc"][0, 32 + channel] = 1.0

    return p
