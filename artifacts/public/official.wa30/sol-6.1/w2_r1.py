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
    p = {name: np.zeros(shape, np.float32)
         for name, shape in shapes.items()}

    def c1(out, color, dy=0, dx=0, weight=1.0):
        p["W1"][out, color, dy + 2, dx + 2] += weight

    def c2(out, channel, dy=0, dx=0, weight=1.0):
        p["W2"][out, channel, dy + 1, dx + 1] += weight

    def point1(out, colors):
        for color in colors:
            c1(out, color)

    def dilate2(out, channel, weight=1.0):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                c2(out, channel, dy, dx, weight)

    def local3(out, channel, weight=1.0):
        p["W3"][out, channel] += weight

    def row3(out, channel, weight=1.0):
        p["W3"][out, 32 + channel] += weight

    def col3(out, channel, weight=1.0):
        p["W3"][out, 64 + channel] += weight

    inner = ((0, 0), (0, 1), (1, 0), (1, 1))
    corners = ((-1, -1), (-1, 2), (2, -1), (2, 2))

    # First-layer object anchors use the upper-left interior pixel
    # of a four-by-four sprite as their reference position.
    for dy, dx in inner:
        c1(0, 14, dy, dx)
    for dy, dx in corners:
        allowed = (14, 0) if dy == -1 else (14, 0, 7, 4)
        for color in allowed:
            c1(0, color, dy, dx)
    p["b1"][0] = -7.0

    # Available crate anchor: border 4, or border 3 when faced.
    for dy, dx in inner:
        c1(4, 9, dy, dx)
    for dy, dx in corners:
        allowed = (4, 3) if dy == -1 else (4, 3, 7)
        for color in allowed:
            c1(4, color, dy, dx)
    p["b1"][4] = -7.0

    # Player-carried crate anchor: black border, interior 9.
    for dy, dx in inner:
        c1(5, 9, dy, dx)
    for dy, dx in corners:
        allowed = (0,) if dy == -1 else (0, 7, 4)
        for color in allowed:
            c1(5, color, dy, dx)
    p["b1"][5] = -7.0

    point1(7, (15,))       # Thief body.
    point1(8, (11,))       # Thief currently faced by the player.
    point1(9, (3,))        # Crate currently faced by the player.
    bad_colors = (4, 3, 5, 12, 15, 11)
    point1(10, bad_colors)
    point1(11, (5,))
    point1(12, (12,))
    point1(13, (4, 3))

    # Goal-rim evidence. Crate interiors are rejected by their
    # neighboring non-goal border colors.
    c1(15, 9, weight=2.0)
    c1(20, 2, weight=2.0)
    p["b1"][15] = -2.0
    p["b1"][20] = -2.0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            c1(15, 2, dy, dx, 0.1)
            c1(20, 9, dy, dx, 0.1)
            for color in (4, 3, 5):
                c1(15, color, dy, dx, -1.0)

    # Shifted rim evidence expands the axial goal footprint without
    # turning plain red patches into destinations.
    for out, dy, dx, perpendicular in (
        (6, 0, -2, ((-1, 0), (1, 0))),
        (14, 0, 2, ((-1, 0), (1, 0))),
        (18, -2, 0, ((0, -1), (0, 1))),
        (19, 2, 0, ((0, -1), (0, 1))),
    ):
        c1(out, 9, dy, dx, 2.0)
        for py, px in perpendicular:
            c1(out, 2, dy + py, dx + px, 0.5)
        p["b1"][out] = -2.0

    # Exact patterned staging-tile detectors at the two interior
    # reference pixels. Transparency exposes background color 1.
    pattern = (
        (2, 1, 2, 2),
        (1, 2, 2, 2),
        (2, 2, 2, 1),
        (2, 2, 1, 2),
    )
    for out, anchor in ((16, 1), (17, 2)):
        for y in range(4):
            for x in range(4):
                c1(out, pattern[y][x], y - anchor, x - anchor)
        p["b1"][out] = -15.0

    # Neighbor occupancy probes. The asymmetric distances account
    # for the upper-left interior reference pixel.
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for a, (dy, dx) in enumerate(directions):
        for color in bad_colors:
            c1(23 + a, color, 2 * dy, 2 * dx)

    # Destination probes for a one-tile translation. A goal has
    # border 9; a patterned tile has its characteristic 2/1 edge.
    c1(27, 9, -2, 0, 2.0)
    c1(27, 2, -2, 0)
    c1(27, 1, -2, 1)
    p["b1"][27] = -1.0

    c1(28, 9, 2, 0, 2.0)
    c1(28, 1, 2, 0)
    c1(28, 2, 2, -1)
    p["b1"][28] = -1.0

    c1(29, 9, 0, -2, 2.0)
    c1(29, 2, 0, -2)
    c1(29, 1, 1, -2)
    p["b1"][29] = -1.0

    c1(30, 9, 0, 2, 2.0)
    c1(30, 1, 0, 2)
    c1(30, 2, -1, 2)
    p["b1"][30] = -1.0

    # Second-layer feature routing.
    c2(0, 0)              # Player anchor.
    c2(1, 5)              # Carried crate anchor.
    c2(2, 4)              # Available crate anchor.

    shifts = ((0, 0), (1, 0), (0, 0), (0, 1))
    for a, (dy, dx) in enumerate(shifts):
        c2(3 + a, 23 + a, dy, dx)
        c2(28 + a, 27 + a, dy, dx)

    # Seeking can approach available crates, but not walls,
    # helpers, thieves, or patterned player-forbidden terrain.
    for channel in (11, 12, 7, 8):
        c2(7, channel)
    for channel in (16, 17):
        dilate2(7, channel)

    # Delivery field excludes the carried black-bordered crate.
    c2(8, 10)

    # Thief field permits approach to thieves themselves.
    for channel in (11, 12, 13):
        c2(9, channel)
    for channel in (16, 17):
        dilate2(9, channel)

    c2(10, 15)
    c2(10, 20)
    for channel in (16, 17):
        dilate2(11, channel)

    # A paired ReLU implements a clipped expanded goal footprint.
    # Axial maxima commute with this monotone clipping operation.
    for channel in (6, 14, 18, 19, 15, 20):
        dilate2(12, channel, 2.0)
        dilate2(24, channel, 2.0)
    p["b2"][24] = -1.0

    c2(13, 7)
    c2(14, 8)
    c2(15, 9)
    c2(23, 12)

    # Third-layer spatial features.
    local3(0, 2)
    for axis in (row3, col3):
        axis(0, 12, -0.6)
        axis(0, 24, 0.6)

    local3(1, 7)
    local3(2, 10)
    local3(2, 11, 0.5)
    local3(3, 8)
    local3(4, 13)
    local3(5, 9)
    local3(6, 0)
    local3(7, 1)

    for a in range(4):
        # Invalid player and carried-crate translations.
        local3(8 + a, 0)
        local3(8 + a, 3 + a)
        p["b3"][8 + a] = -1.0

        local3(12 + a, 1)
        local3(12 + a, 3 + a)
        p["b3"][12 + a] = -1.0

        # Predicted delivery-tile arrivals.
        local3(16 + a, 1)
        local3(16 + a, 28 + a)
        p["b3"][16 + a] = -1.0

    local3(20, 15)
    local3(21, 14)
    local3(22, 1)
    local3(23, 23)
    local3(24, 11, 0.5)
    local3(29, 10)
    local3(30, 2)

    # Three screened-diffusion fields. Their neighbor coefficient
    # sums to 0.96, keeping the iteration bounded and preserving
    # spatial gradients after 128 iterations.
    for field, source, obstacle in (
        (0, 0, 1),        # Available crates.
        (1, 2, 3),        # Goals or patterned staging tiles.
        (2, 4, 5),        # Thieves.
    ):
        p["Uh"][field, source] = 1.0
        p["Uh"][field, obstacle] = -1000.0
        for dy, dx in directions:
            p["Wh"][field, field, dy + 1, dx + 1] = 0.24

    # Twelve gated directional readouts: four per field.
    for task in range(3):
        anchor = 7 if task == 1 else 6
        for a, (dy, dx) in enumerate(directions):
            out = 3 + 4 * task + a
            p["Wh"][out, task, dy + 1, dx + 1] = 1.0
            p["bh"][out] = -1000.0
            p["Uh"][out, anchor] = 1000.0
            if task == 1:
                p["Uh"][out, 12 + a] = -1000.0

    # Global feature indices in g.
    def fmax(channel):
        return 32 + channel

    # Action-score units.
    for a in range(4):
        p["Wz"][a, 3 + a] = 100.0
        p["Wz"][a, fmax(22)] = -10000.0
        p["Wz"][a, fmax(4)] = -10000.0

        p["Wz"][4 + a, 7 + a] = 100.0
        p["Wz"][4 + a, fmax(8 + a)] = -10000.0

        p["Wz"][8 + a, 11 + a] = 100.0
        p["Wz"][8 + a, fmax(22)] = -10000.0

        for unit in (a, 4 + a, 8 + a):
            p["Wo"][1 + a, unit] = 1.0

    # Attach a faced crate only when not carrying or pursuing thieves.
    p["Wz"][12, fmax(20)] = 1.0
    p["Wz"][12, fmax(22)] = -1.0
    p["Wz"][12, fmax(4)] = -1.0
    p["Wo"][5, 12] = 30000.0

    # ACTION5 removes a faced thief.
    p["Wz"][13, fmax(21)] = 1.0
    p["Wo"][5, 13] = 30000.0

    # Store four valid predicted-arrival flags. At the following
    # observation, combine each flag with the actual previous action.
    for a in range(4):
        unit = 14 + a
        p["Wz"][unit, fmax(16 + a)] = 1.0
        p["Wz"][unit, fmax(8 + a)] = -1.0

        p["Wmz"][a, unit] = 20.0
        p["bm"][a] = -10.0

        p["Wm"][4 + a, a] = 10.0
        p["Wma"][4 + a, 1 + a] = 20.0
        p["bm"][4 + a] = -20.0
        p["Wo"][5, 128 + 4 + a] = 10000.0

    # The four inactive arrival units are exactly -1 in float32;
    # this bias cancels them, making ACTION5 the harmless wait action
    # when no navigation or interaction score is positive.
    p["bo"][5] = 40000.0
    p["bo"][1:5] = -1e-30
    p["bo"][0] = -1000000.0
    p["bo"][6:8] = -1000000.0

    return p
