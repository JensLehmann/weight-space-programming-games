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

    # Up, down, left, right.
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    quadrants = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    def c1(out, inp, dy=0, dx=0, value=1.0):
        p["W1"][out, inp, dy + 2, dx + 2] += value

    def c2(out, inp, dy=0, dx=0, value=1.0):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    def ch(out, inp, dy=0, dx=0, value=1.0):
        p["Wh"][out, inp, dy + 1, dx + 1] += value

    def mix(out, terms, bias=0.0):
        for inp, value in terms:
            p["W3"][out, inp] += value
        p["b3"][out] = bias

    # ------------------------------------------------------------
    # First layer: materials, markers, outlines, contact geometry.
    # ------------------------------------------------------------

    c1(0, 2)                 # Wall.
    c1(1, 15)                # Gulch.

    for dy in range(-2, 3):
        for dx in range(-2, 3):
            c1(2, 2, dy, dx)  # Wider wall-clearance test.

    for color in (11, 12, 13):
        c1(3, color)         # Cross / bomb material.

    # Active central marker. The cardinal test rejects ordinary
    # black contact-edge pixels.
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            c1(4, 0, dy, dx, 10.0)
            c1(4, 14, dy, dx, 10.0)
    c1(4, 0, value=10.0)
    for dy, dx in dirs:
        c1(4, 14, dy, dx)
    p["b1"][4] = -101.0

    # Inactive central markers, including the larger controllers.
    c1(5, 4, value=10.0)
    c1(5, 5, value=10.0)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            c1(5, 14, dy, dx)
    p["b1"][5] = -13.0

    # Small rectangular goal:
    # all four corners must be outline or controller material;
    # at least three must still be visible outline.
    for dy in (-2, 2):
        for dx in (-2, 2):
            c1(6, 4, dy, dx, 11.0)
            c1(6, 14, dy, dx, 10.0)
            c1(6, 0, dy, dx, 10.0)
    p["b1"][6] = -42.0

    # Concave corners of the cross-shaped goal.
    for j, (qy, qx) in enumerate(quadrants):
        out = 7 + j
        for dy, dx in (
            (qy, qx),
            (qy, 2 * qx),
            (2 * qy, qx),
        ):
            c1(out, 4, dy, dx)
        for dy, dx in (
            (qy, 0),
            (0, qx),
            (2 * qy, 2 * qx),
        ):
            c1(out, 4, dy, dx, -1.0)
        p["b1"][out] = -2.0

    for i, (dy, dx) in enumerate(dirs):
        py, px = -dx, dy

        # Contact with a cross arm, viewed from a free staging cell.
        for t in (-1, 0, 1):
            c1(11 + i, 11, 2 * dy + t * py, 2 * dx + t * px)
        c1(11 + i, 11, dy, dx, -1.0)
        p["b1"][11 + i] = -2.0

        # Contact with an inactive small controller.
        for t in (-1, 0, 1):
            c1(15 + i, 14, 2 * dy + t * py, 2 * dx + t * px)
        # Do not routinely kick a controller out of a completed goal.
        c1(15 + i, 4, dy, dx, -1.0)
        p["b1"][15 + i] = -2.0

        # Marker sampled two pixels forward; W2 supplies pixel three.
        c1(19 + i, 5, 2 * dy, 2 * dx)
        c1(19 + i, 4, 2 * dy, 2 * dx)

        # Forward wall/gulch strip.
        for distance in (1, 2):
            for t in (-1, 0, 1):
                for color in (2, 15):
                    c1(
                        28 + i, color,
                        distance * dy + t * py,
                        distance * dx + t * px,
                    )

    # Non-selected controller material. Nearby black markers exempt
    # the selected controller from its own navigation obstacle mask.
    c1(23, 14)
    c1(23, 5)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            c1(23, 0, dy, dx, -1.0)

    # Occupied small-controller center, for suppressing filled goals.
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy or dx:
                c1(24, 14, dy, dx)
                c1(24, 0, dy, dx)
    for color in (0, 4, 5):
        c1(24, color)
    p["b1"][24] = -8.0

    c1(25, 14)              # Reachability seeds from every controller.

    # A simple upward bomb-riding staging predicate.
    for dx in (-1, 0, 1):
        for color in (12, 13):
            c1(26, color, 2, dx)
    for color in (12, 13):
        c1(26, color, 1, 0, -1.0)
        c1(27, color)
    p["b1"][26] = -2.0

    # ------------------------------------------------------------
    # Second layer: clearance and completed local predicates.
    # ------------------------------------------------------------

    for dy in range(-1, 2):
        for dx in range(-1, 2):
            c2(0, 0, dy, dx)       # Small-piece wall obstruction.

            for inp in (0, 1, 3, 23):
                c2(1, inp, dy, dx) # Walking obstruction.

            c2(2, 2, dy, dx)       # Approximate cross obstruction.

            for inp in (0, 1):
                c2(18, inp, dy, dx)  # Reachability obstruction.

    c2(3, 4, value=-1.0)
    p["b2"][3] = 1.0        # Zero at an active central marker.

    c2(4, 5, value=-1.0)
    p["b2"][4] = 1.0        # Zero at an inactive central marker.

    # Reject the extended plateau generated by a wider rectangular
    # goal's horizontal edges.
    c2(5, 6, value=-1.0)
    for dy, dx in dirs:
        c2(5, 6, dy, dx)
    p["b2"][5] = 1.0

    for j, (qy, qx) in enumerate(quadrants):
        c2(6, 7 + j, qy, qx)
        c2(7, 7 + j, qy, qx)
    p["b2"][6] = -1.0
    c2(7, 3)
    p["b2"][7] = -4.0

    for i, (dy, dx) in enumerate(dirs):
        c2(8 + i, 11 + i)
        c2(8 + i, 28 + i, dy, dx, -1.0)

        c2(12 + i, 15 + i)
        c2(12 + i, 19 + i, dy, dx)
        c2(12 + i, 28 + i, dy, dx, -1.0)
        p["b2"][12 + i] = -1.0

    c2(16, 24)
    c2(17, 25)
    c2(19, 27)
    c2(20, 26)

    # ------------------------------------------------------------
    # Pointwise features. Obstruction predicates use zero for valid
    # cells, avoiding cancellation of small propagation signals.
    # ------------------------------------------------------------

    mix(0, [(0, 1.0)])       # Small-piece wall obstruction.
    mix(1, [(1, 1.0)])       # Walking obstruction.
    mix(2, [(2, 1.0)])       # Cross obstruction.

    mix(3, [(5, -1.0), (16, -1.0)], 1.0)  # Empty small goal.
    mix(4, [(6, 1.0), (7, -3.0)])          # Empty cross goal.

    for i in range(4):
        terms = [(12 + i, -1.0)]
        if i == 0:
            terms.append((20, -1.0))
        mix(5 + i, terms, 1.0)

    # Canonical cross-goal layouts predominantly require upward and
    # horizontal kicks. Those three staging directions have channels.
    cross_dirs = (0, 2, 3)
    for j, i in enumerate(cross_dirs):
        mix(9 + j, [(8 + i, -1.0)], 1.0)

    mix(12, [(3, 1.0)])      # Active-marker forbidden mask.
    mix(13, [(4, 1.0)])      # Inactive-marker forbidden mask.
    mix(14, [(17, 1.0)])     # All-controller reachability seeds.
    mix(15, [(18, 1.0)])     # Reachability obstruction.
    mix(16, [(19, 1.0)])     # Bomb presence.
    mix(17, [(5, -1.0), (4, -1.0)], 1.0)  # Filled inactive goal.

    # ------------------------------------------------------------
    # Recurrent block.
    #
    # 0,1: clipped binary reachability flood, represented by h0-h1.
    # 2:   goals outside that reachability flood.
    # 3:   remote rectangular-goal potential.
    # 4:   cross-goal potential.
    # 5:8: controller/bomb staging scores, U,D,L,R.
    # 9:11: cross staging scores, U,L,R.
    # 12:  walking/navigation potential.
    # 13:15: selected-piece sum, horizontal, vertical readouts.
    # ------------------------------------------------------------

    gate = 1.0e30

    # Binary reachability: relu(v) - relu(v-1).
    # Including the center makes reached cells persistent.
    for out in (0, 1):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ch(out, 0, dy, dx, 1.0)
                ch(out, 1, dy, dx, -1.0)
        p["Uh"][out, 14] = 1.0
        p["Uh"][out, 15] = -32.0
    p["bh"][1] = -1.0

    p["Uh"][2, 3] = 1.0
    ch(2, 0, value=-1.0)
    ch(2, 1, value=1.0)

    for dy, dx in dirs:
        ch(3, 3, dy, dx, 0.20)
        ch(4, 4, dy, dx, 0.249)
        ch(12, 12, dy, dx, 0.22)

    ch(3, 2)
    p["Uh"][3, 0] = -gate

    p["Uh"][4, 4] = 1.0
    p["Uh"][4, 2] = -gate

    rectangular_gain = 1.0e15
    cross_gain = 1.0e12

    for i, (dy, dx) in enumerate(dirs):
        out = 5 + i
        ch(out, 3, dy, dx, 1.0)
        ch(out, 3, -dy, -dx, -1.0)
        p["Uh"][out, 5 + i] = -gate
        p["Uh"][out, 1] = -gate
        # Remove late residuals from the initial reachability transient.
        p["bh"][out] = -1.0e-15
        ch(12, out, value=rectangular_gain)

    for j, i in enumerate(cross_dirs):
        dy, dx = dirs[i]
        out = 9 + j
        ch(out, 4, dy, dx, 1.0)
        ch(out, 4, -dy, -dx, -1.0)
        p["Uh"][out, 9 + j] = -gate
        p["Uh"][out, 1] = -gate
        p["bh"][out] = -1.0e-30
        ch(12, out, value=cross_gain)

    p["Uh"][12, 3] = 1.0
    p["Uh"][12, 1] = -gate

    # Three nonnegative readouts encode both signed navigation axes.
    # S = U+D+L+R
    # X = U+D+2R
    # Y = L+R+2D
    # Thus R-L = X-S and D-U = Y-S.
    for dy, dx in dirs:
        ch(13, 12, dy, dx)

    for i, weight in ((0, 1.0), (1, 1.0), (3, 2.0)):
        dy, dx = dirs[i]
        ch(14, 12, dy, dx, weight)

    for i, weight in ((1, 2.0), (2, 1.0), (3, 1.0)):
        dy, dx = dirs[i]
        ch(15, 12, dy, dx, weight)

    staging = [
        (5 + i, i, rectangular_gain) for i in range(4)
    ] + [
        (9 + j, i, cross_gain) for j, i in enumerate(cross_dirs)
    ]

    for channel, direction, gain in staging:
        force = 10.0 * gain
        ch(13, channel, value=force)

        x_weight = (1.0, 1.0, 0.0, 2.0)[direction]
        y_weight = (0.0, 2.0, 1.0, 1.0)[direction]
        ch(14, channel, value=force * x_weight)
        ch(15, channel, value=force * y_weight)

    for out in (13, 14, 15):
        p["Uh"][out, 12] = -gate

    # ------------------------------------------------------------
    # Action and click readout.
    # ------------------------------------------------------------

    # Spatial sums preserve the linear axis identities even when a
    # larger selected marker occupies more than one central pixel.
    for unit, channel in enumerate((13, 14, 15)):
        p["Wz"][unit, 16 + channel] = 4096.0

    p["Wo"][1, 0] = 1.0
    p["Wo"][1, 2] = -1.0
    p["Wo"][2, 2] = 1.0
    p["Wo"][2, 0] = -1.0
    p["Wo"][3, 0] = 1.0
    p["Wo"][3, 1] = -1.0
    p["Wo"][4, 1] = 1.0
    p["Wo"][4, 0] = -1.0

    # Switch only when navigation has no positive directional signal.
    p["bo"][6] = 1.0e-30

    # If a bomb is present, a blocked upward move advances its timer,
    # unlike repeated clicks.
    p["Wz"][3, 32 + 16] = 1.0
    p["Wo"][1, 3] = 2.0e-30

    p["bo"][0] = -1.0e30
    p["bo"][5] = -1.0e30
    p["bo"][7] = -1.0e30

    # Mask the click map directly: no additional recurrent channel
    # is necessary. Prefer unfinished inactive controllers.
    p["Wc"][0, 13] = -gate
    p["Wc"][0, 17] = -1.0
    p["Wc"][0, 32 + 12] = 1.0
    p["Wc"][0, 32 + 3] = 1.0

    return p
