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
    p = {k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()}
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    marker = (4, 5)
    bomb = (12, 13)

    def w1(o, colors, y=0, x=0, v=1.0):
        if isinstance(colors, int):
            colors = (colors,)
        for c in colors:
            p["W1"][o, c, y + 2, x + 2] += v

    def w2(o, c, y=0, x=0, v=1.0):
        p["W2"][o, c, y + 1, x + 1] += v

    def w3(o, c, v=1.0, pool=0):
        p["W3"][o, c + 32 * pool] += v

    def wh(o, c, y=0, x=0, v=1.0):
        p["Wh"][o, c, y + 1, x + 1] += v

    # ------------------------------------------------------------
    # First convolution: obstacles, cores, frame corners and faces.
    # ------------------------------------------------------------

    # Three-by-three footprint obstacles, including image boundaries.
    for o in (0, 1):
        p["b1"][o] = 9.0
        for y in range(-1, 2):
            for x in range(-1, 2):
                w1(o, 18, y, x, -1.0)
                w1(o, 2, y, x)
                if o == 1:
                    w1(o, 15, y, x)

    # Capped selected-core detector. Contact edges may also be black,
    # but do not have a full piece neighborhood and two border neighbors.
    for o in (2, 29):
        for y in range(-1, 2):
            for x in range(-1, 2):
                w1(o, (0, 14), y, x, 10.0)
        w1(o, 0, v=100.0)
        for y, x in dirs:
            w1(o, 14, y, x)
    p["b1"][2] = -191.0
    p["b1"][29] = -192.0

    # Upper-left marker pixel of an unselected piece.
    for y in range(-1, 2):
        for x in range(-1, 2):
            w1(3, (4, 5, 14), y, x)
    w1(3, marker, v=10.0)
    w1(3, marker, -1, 0, -20.0)
    w1(3, marker, 0, -1, -20.0)
    p["b1"][3] = -18.0

    # A convex frame corner has a frame pixel and no outward frame
    # continuation. Unlike inward-leg tests, these tolerate covered sides.
    corners = [
        (4, -1, -1),
        (25, -1, 1),
        (26, 1, -1),
        (27, 1, 1),
    ]
    for c, sy, sx in corners:
        w1(c, 4, sy, sx)
        w1(c, 4, 2 * sy, sx, -1.0)
        w1(c, 4, sy, 2 * sx, -1.0)

    # Vertical rectangular goal, upper portion.
    for x in range(-2, 3):
        w1(5, 4, -2, x)
    for y in range(-1, 3):
        w1(5, 4, y, -2)
        w1(5, 4, y, 2)
        for x in range(-1, 2):
            w1(5, 1, y, x)
    p["b1"][5] = -24.0

    # Horizontal rectangular goal, left portion.
    for y in range(-2, 3):
        w1(6, 4, y, -2)
    for x in range(-1, 3):
        w1(6, 4, -2, x)
        w1(6, 4, 2, x)
        for y in range(-1, 2):
            w1(6, 1, y, x)
    p["b1"][6] = -24.0

    # Large square goal, upper-left portion.
    for x in range(-2, 3):
        w1(7, 4, -2, x)
    for y in range(-1, 3):
        w1(7, 4, y, -2)
    for y in range(-1, 3):
        for x in range(-1, 3):
            w1(7, 1, y, x)
    p["b1"][7] = -24.0

    # Cross goal.
    for y in (-2, 2):
        for x in (-2, 2):
            w1(8, 4, y, x)
    for y, x in ((0, 0), (-2, 0), (2, 0), (0, -2), (0, 2)):
        w1(8, 1, y, x)
    p["b1"][8] = -8.0

    for d, (dy, dx) in enumerate(dirs):
        ty, tx = dx, -dy

        # Rear boundary for launching in direction d.
        # Narrow rectangular faces give 1; wide faces give 2.
        o = 9 + d
        w1(o, 14, v=8.0)
        w1(o, (11, 12, 13), v=10.0)
        w1(o, marker, dy, dx, 2.0)
        w1(o, marker, 2 * dy, 2 * dx, 2.0)
        w1(o, (11, 12, 13, 14), -dy, -dx, -10.0)
        w1(o, 14, 2 * ty, 2 * tx, 0.5)
        w1(o, 14, -2 * ty, -2 * tx, 0.5)
        p["b1"][o] = -9.0

        # Bomb outgoing boundary.
        w1(13 + d, bomb)
        w1(13 + d, bomb, dy, dx, -1.0)

        # Wall-contact / bomb-filling directional cue.
        o = 17 + d
        w1(o, (11, 13, 14), v=2.0)
        w1(o, 12, -dy, -dx)
        w1(o, 2, dy, dx)
        w1(o, 2, 2 * dy, 2 * dx)
        p["b1"][o] = -2.0

    w1(21, 16)
    w1(22, 17)

    # Bottom interface row.
    w1(23, 16, v=63.0)
    p["b1"][23] = -62.0

    # Occupancy anchor, independent of the interior color.
    for y, x in ((-2, -2), (-2, 0), (0, -2)):
        w1(24, 4, y, x)
    p["b1"][24] = -2.0

    w1(28, 4)

    # Two optional shape features for click ordering.
    w1(30, marker)
    w1(30, marker, 0, 1)
    w1(30, marker, -1, 0, -1.0)
    w1(30, marker, 1, 0, -1.0)
    p["b1"][30] = -1.0

    for y, x in ((0, 0), (0, 1), (1, 0), (1, 1)):
        w1(31, marker, y, x)
    p["b1"][31] = -3.0

    # ------------------------------------------------------------
    # Second convolution: align and cap the spatial predicates.
    # ------------------------------------------------------------

    for o in (0, 1):
        w2(o, o)
        for y in range(-1, 2):
            for x in range(-1, 2):
                w2(o, 23, y, x)

    w2(2, 2)
    w2(2, 29, v=-1.0)
    w2(3, 3)

    # Ordinary goal = three surviving convex corners, or a recognized
    # larger frame. The pair difference is strictly capped at one.
    for o in (4, 5):
        for c, sy, sx in corners:
            w2(o, c, sy, sx)
        w2(o, 5, -1, 0, 3.0)
        w2(o, 6, 0, -1, 3.0)
        w2(o, 7, -1, -1, 3.0)
        w2(o, 3, v=-10.0)
        w2(o, 2, v=-10.0)
        w2(o, 29, v=10.0)
    p["b2"][4] = -2.0
    p["b2"][5] = -3.0

    for d, (dy, dx) in enumerate(dirs):
        ty, tx = dx, -dy
        ca, cb = 6 + 2 * d, 7 + 2 * d
        for o in (ca, cb):
            w2(o, 9 + d, dy, dx, 2.0)
            w2(o, 9 + d, dy + ty, dx + tx)
            w2(o, 9 + d, dy - ty, dx - tx)
            w2(o, 28, v=-10.0)
        p["b2"][ca] = -1.0
        p["b2"][cb] = -2.0

        w2(14 + d, 13 + d, -dy, -dx)
        w2(14 + d, 28, v=-10.0)
        w2(18 + d, 17 + d)

    # Selected coordinates.
    w2(22, 21)
    w2(22, 2, v=2.0)
    w2(22, 29, v=-2.0)
    p["b2"][22] = -2.0

    w2(23, 22)
    w2(23, 2, v=2.0)
    w2(23, 29, v=-2.0)
    p["b2"][23] = -2.0

    # Capped selected/unselected goal occupancy.
    offsets = ((0, 0), (-1, 0), (0, -1), (-1, -1))
    for ca, cb, core in ((24, 25, 2), (26, 27, 3)):
        for o in (ca, cb):
            w2(o, core, v=4.0)
            if core == 2:
                w2(o, 29, v=-4.0)
            for y, x in offsets:
                w2(o, 24, y, x)
        p["b2"][ca] = -4.0
        p["b2"][cb] = -5.0

    w2(28, 8)
    for o, c in ((29, 30), (30, 31)):
        w2(o, c)
        w2(o, 3)
        p["b2"][o] = -1.0

    # Unselected-object neighborhood.
    for y in range(-1, 2):
        for x in range(-1, 2):
            w2(31, 3, y, x)

    # ------------------------------------------------------------
    # Axial layer.
    # ------------------------------------------------------------

    w3(0, 0)
    w3(1, 1)
    w3(2, 2)

    w3(3, 3)
    w3(3, 26, -1.0)
    w3(3, 27, 1.0)

    w3(4, 4)
    w3(4, 5, -1.0)
    w3(5, 4)
    w3(5, 5, -1.0)
    w3(5, 28)

    for d, (dy, dx) in enumerate(dirs):
        pool = 2 if dy else 1
        w3(6 + d, 6 + 2 * d)
        w3(6 + d, 7 + 2 * d, -1.0)
        w3(6 + d, 26, -1.0, pool)
        w3(6 + d, 27, 1.0, pool)
        w3(6 + d, 18 + d, -1.0, pool)

        # Strong local gating prevents an axial cue by itself from
        # creating spurious bomb mounting sites.
        w3(10 + d, 14 + d, 10.0)
        w3(10 + d, 18 + d, 1.0, pool)
        p["b3"][10 + d] = -10.0

    w3(14, 22)
    w3(15, 23)
    w3(16, 24)
    w3(16, 25, -1.0)
    w3(17, 3)

    for o, c in ((18, 29), (19, 30)):
        w3(o, c)
        w3(o, 26, -1.0)
        w3(o, 27, 1.0)

    w3(22, 2)
    w3(22, 3)
    w3(23, 28)
    w3(24, 1)
    w3(24, 31, 100.0)

    # ------------------------------------------------------------
    # Recurrent planner.
    # Pair differences implement capped binary wavefronts.
    # ------------------------------------------------------------

    def neighbors(o, ca, cb):
        for dy, dx in dirs:
            wh(o, ca, dy, dx)
            wh(o, cb, dy, dx, -1.0)

    # 0/1: ordinary-walking components containing controllers.
    for o in (0, 1):
        neighbors(o, 0, 1)
        p["Uh"][o, 22] = 1.0
        p["Uh"][o, 1] = -10000.0
        p["Uh"][o, 23] = -10000.0
    p["bh"][1] = -1.0

    # 4: clocks at goals not served by a controller component.
    wh(4, 4)
    wh(4, 0, v=-1000.0)
    wh(4, 1, v=1000.0)
    p["Uh"][4, 5] = 1000.0
    p["bh"][4] = -999.0

    # 2/3: launch-planning field, allowed to cross gulches.
    # Rectangular deliveries precede cross positioning.
    for o in (2, 3):
        neighbors(o, 2, 3)
        wh(o, 4)
        p["Uh"][o, 0] = -10000.0
        p["Uh"][o, 5] = -32.0
        p["Uh"][o, 23] = -32.0
    p["bh"][3] = -1.0

    # 5: clocks at ordinary goals.
    wh(5, 5)
    p["Uh"][5, 4] = 1000.0
    p["bh"][5] = -999.0

    # 6..9: latch useful launch directions from wavefront arrival order.
    for d, (dy, dx) in enumerate(dirs):
        o = 6 + d
        wh(o, o)
        wh(o, 2, dy, dx)
        wh(o, 3, dy, dx, -1.0)
        wh(o, 2, v=-1.0)
        wh(o, 3, v=1.0)
        p["Uh"][o, 6 + d] = 2.0
        p["Uh"][o, 1] = -10000.0
        p["bh"][o] = -2.0

    # 10/11: controller walking field.
    # A face indicator is one pixel outside the object. Shift its
    # navigation source one more pixel to obtain a controller center.
    for o in (10, 11):
        neighbors(o, 10, 11)
        for d, (dy, dx) in enumerate(dirs):
            wh(o, 6 + d, dy, dx)
        wh(o, 5)
        p["Uh"][o, 4] = -100.0
        p["Uh"][o, 24] = -10000.0
        for c in range(10, 14):
            p["Uh"][o, c] = 1.0
    p["bh"][11] = -1.0

    # 12: accumulated arrival-time potential.
    wh(12, 12)
    wh(12, 10)
    wh(12, 11, v=-1.0)

    # 13..15: action advantages relative to ACTION4.
    for d in range(3):
        o = 13 + d
        dy, dx = dirs[d]
        wh(o, 12, dy, dx)
        wh(o, 12, 0, 1, -1.0)
        wh(o, 10, v=1000.0)
        wh(o, 11, v=-1000.0)
        wh(o, 6 + d, dy, dx, 8.0)
        wh(o, 9, 0, 1, -8.0)
        p["Uh"][o, 2] = 65536.0
        p["bh"][o] = -65536.0 + 128.0

    # ------------------------------------------------------------
    # Global action readout.
    # ------------------------------------------------------------

    for d in range(3):
        p["Wz"][d, 13 + d] = 1.0

    # Difference z3-z4 is selected-core reachability.
    p["Wz"][3, 13] = 1.0
    p["bz"][3] = -500.0
    p["Wz"][4, 13] = 1.0
    p["bz"][4] = -501.0

    p["Wz"][5, 32 + 16] = 1.0
    p["Wz"][6, 32 + 17] = 1.0
    p["Wz"][16, 32 + 14] = 1.0
    p["Wz"][17, 32 + 15] = 1.0

    p["bo"][:] = -1.0e6
    for d in range(3):
        p["bo"][d + 1] = 0.0
        p["Wo"][d + 1, d] = 1.0

    p["bo"][4] = 128.0
    p["Wo"][4, 3] = 1000.0
    p["Wo"][4, 4] = -1000.0

    p["bo"][6] = -700.0
    p["Wo"][6, 6] = 1000.0
    p["Wo"][6, 3] = -2000.0
    p["Wo"][6, 4] = 2000.0
    p["Wo"][6, 5] = 4096.0

    # ------------------------------------------------------------
    # Memory: detect stationary repeats from selected coordinates.
    # ------------------------------------------------------------

    scale = 0.02
    p["Wmz"][0, 16] = scale
    p["Wmz"][1, 17] = scale

    gain = 200000.0
    tolerance = 0.00005
    gate = 10000.0
    penalty = 4.0

    for d in range(4):
        action = d + 1
        base = 2 + 5 * d
        for axis in range(2):
            for side in range(2):
                unit = base + 2 * axis + side
                sign = 1.0 if side == 0 else -1.0
                p["Wm"][unit, axis] = gain
                p["Wmz"][unit, 16 + axis] = -gain * scale
                p["Wma"][unit, action] = -gate
                p["bm"][unit] = gate + sign * gain * tolerance
                p["Wo"][action, 128 + unit] = (
                    -penalty if side == 0 else penalty
                )
        unit = base + 4
        p["Wma"][unit, action] = 10.0
        p["Wo"][action, 128 + unit] = 2.0 * penalty

    # Click map: unfinished cores first, then shape and spatial priority.
    p["Wc"][0, 3] = 1000.0
    p["Wc"][0, 17] = 1000.0
    p["Wc"][0, 18] = 400.0
    p["Wc"][0, 19] = 200.0
    p["Wc"][0, 32 + 12] = 0.1

    return p
