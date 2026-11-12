import math
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
    p = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in shapes.items()
    }

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    sizes = [(3, 3), (3, 6), (6, 3), (6, 6)]
    offsets = [(1, 1), (1, 2), (2, 1), (2, 2)]

    def c1(out, source, dy=0, dx=0, weight=1.0):
        p["W1"][out, source, dy + 2, dx + 2] += weight

    def colours(out, sources, dy=0, dx=0, weight=1.0):
        for source in sources:
            c1(out, source, dy, dx, weight)

    def square(out, sources, weight=1.0):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                colours(out, sources, dy, dx, weight)

    def c2(out, source, dy=0, dx=0, weight=1.0):
        p["W2"][out, source, dy + 1, dx + 1] += weight

    def c3(out, source, weight=1.0):
        p["W3"][out, source] += weight

    def ch(out, source, dy=0, dx=0, weight=1.0):
        p["Wh"][out, source, dy + 1, dx + 1] += weight

    def selected(out, dy=0, dx=0):
        c1(out, 0, dy, dx, 10.0)
        for sy, sx in directions:
            colours(out, (0, 14), dy + sy, dx + sx)
        p["b1"][out] = -13.0

    # -----------------------------------------------------------
    # First layer: local geometry.
    # -----------------------------------------------------------

    square(0, (2,))
    square(1, (15,))

    # Inactive rectangular interiors. Revisited interiors use colour 4.
    colours(2, (4, 5), weight=10.0 / 3.0)
    for dy, dx in directions:
        c1(2, 14, dy, dx, 1.0 / 3.0)
    p["b1"][2] = -11.0 / 3.0
    selected(3)

    # A true frame corner must remain colour 4. Its straight arms may
    # be overwritten by the edge of a touching brick.
    c1(4, 4, -2, -2, 10.0)
    for dy, dx in [(-2, -1), (-1, -2), (-2, 0), (0, -2)]:
        colours(4, (0, 4, 14), dy, dx)
    colours(4, (4, 14), -1, -1, -10.0)
    p["b1"][4] = -13.0

    for k, (width, height) in enumerate(sizes):
        ox, oy = offsets[k]
        colours(5 + 2 * k, (0, 4, 14), -oy, 3 - ox)
        colours(6 + 2 * k, (0, 4, 14), 3 - oy, -ox)

    square(13, (0, 4, 5, 14))
    p["b1"][13] = -8.0

    # Cross-frame marker.
    for dy in (-2, 2):
        for dx in (-2, 2):
            c1(14, 4, dy, dx)
    for dy, dx in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        c1(14, 4, dy, dx, -4.0)
    p["b1"][14] = -3.0

    # Cross occupancy is used by the same centre-gradient mechanism.
    c1(19, 11)

    # Displaced selected masks for reading three-pixel destinations.
    for k, (dy, dx) in enumerate(directions):
        selected(23 + k, -dy, -dx)

    # Checksum inputs and coordinates.
    colours(27, (5, 11, 12, 13, 14))
    c1(28, 16)
    c1(29, 17)

    colours(30, (0, 4, 14), 1, -2)
    colours(31, (0, 4, 14), -2, 1)

    # -----------------------------------------------------------
    # Second layer: clipped terrain, target markers, selected size.
    # -----------------------------------------------------------

    c2(0, 0)
    c2(1, 0)
    p["b2"][1] = -1.0

    for out in (2, 3):
        c2(out, 0)
        c2(out, 1)
    p["b2"][3] = -1.0
    c2(4, 3)

    def target(out, kind, unfinished):
        width, height = sizes[kind]
        ox, oy = offsets[kind]
        c2(out, 4, 1 - oy, 1 - ox)
        c2(
            out, 5 + 2 * kind,
            weight=1.0 if width == 3 else -1.0
        )
        c2(
            out, 6 + 2 * kind,
            weight=1.0 if height == 3 else -1.0
        )
        c2(out, 30, 2 - oy, 1 - ox)
        c2(out, 31, 1 - oy, 2 - ox)
        p["b2"][out] = -(
            int(width == 3) + int(height == 3) + 2
        )
        if unfinished:
            c2(out, 13, weight=-2.0)

    for kind in range(4):
        target(5 + kind, kind, True)
        target(27 + kind, kind, False)
    c2(9, 14)

    # Raw/excess pairs clip the movable-centre mask to one.
    # Suppression is applied at the centre, not at an arbitrary edge.
    for out in (10, 14):
        c2(out, 2, weight=3.0)
        c2(out, 19)
    p["b2"][14] = -1.0

    c2(18, 27)
    c2(19, 28)
    c2(20, 29)
    c2(21, 2)

    c2(22, 3)
    c2(22, 3, -1, 0)
    p["b2"][22] = -1.0

    c2(31, 3)
    c2(31, 3, 0, -1)
    p["b2"][31] = -1.0

    for k, (dy, dx) in enumerate(directions):
        c2(23 + k, 23 + k, -dy, -dx)

    # -----------------------------------------------------------
    # Pointwise layer.
    # -----------------------------------------------------------

    c3(0, 0)
    c3(0, 1, -1.0)
    c3(1, 2)
    c3(1, 3, -1.0)
    c3(2, 4)

    for source in range(5, 9):
        c3(3, source)
        c3(4, source)
    c3(4, 9)

    # Four identical masks gate four differently oriented gradients.
    for out in range(5, 9):
        c3(out, 10)
        c3(out, 14, -1.0)
        for source in range(27, 31):
            c3(out, source, -10.0)

    c3(9, 21, 3.0)
    for source in range(27, 31):
        c3(9, source, -10.0)

    for out, coordinate in ((10, 19), (11, 20)):
        c3(out, coordinate)
        c3(out, 18)
        p["b3"][out] = -1.0

    c3(12, 4)
    for source in range(27, 31):
        c3(12, source)
    p["b3"][12] = -1.0
    c3(13, 21)

    for out, coordinate in ((14, 19), (15, 20)):
        c3(out, coordinate)
        c3(out, 4)
        p["b3"][out] = -1.0

    for k in range(4):
        c3(16 + k, 23 + k)

    # Coordinates of unfinished targets, rather than completed frames.
    for kind in range(4):
        for axis, coordinate in enumerate((19, 20)):
            out = 20 + 2 * kind + axis
            c3(out, 5 + kind)
            c3(out, coordinate)
            p["b3"][out] = -1.0

    c3(28, 31)  # selected interior has two columns
    c3(29, 22)  # selected interior has two rows

    # Presence of unfinished small targets. Its mean also gives count.
    c3(30, 5)

    # -----------------------------------------------------------
    # Recurrent reachability and arrival potentials.
    # -----------------------------------------------------------

    def flood(raw, excess, reached, potential, seed, obstacle):
        for dy, dx in [(0, 0)] + directions:
            ch(raw, reached, dy, dx)
            ch(excess, reached, dy, dx)
        p["Uh"][raw, seed] = 1.0
        p["Uh"][excess, seed] = 1.0
        p["Uh"][raw, obstacle] = -1000.0
        p["Uh"][excess, obstacle] = -1000.0
        p["bh"][excess] = -1.0
        ch(reached, raw)
        ch(reached, excess, weight=-1.0)
        ch(potential, potential)
        ch(potential, reached)

    # Transport field ignores gulches, but respects walls.
    flood(0, 1, 2, 3, 4, 0)

    # Evaluate the outgoing gradient at the object's centre.
    # In particular, a blocked perpendicular exit yields no shove seed.
    for k, (dy, dx) in enumerate(directions):
        out = 4 + k
        ch(out, 3, dy, dx)
        ch(out, 3, -dy, -dx, -1.0)
        p["Uh"][out, 5 + k] = 1000.0
        p["bh"][out] = -1000.0

    # Navigation field. Each shove seed is moved one pixel backward,
    # toward the side from which a helper must approach.
    flood(8, 9, 10, 11, 3, 1)
    for k, (dy, dx) in enumerate(directions):
        ch(8, 4 + k, dy, dx)
        ch(9, 4 + k, dy, dx)

    for k, (dy, dx) in enumerate(directions):
        out = 12 + k
        ch(out, 11, dy, dx)
        ch(out, 4 + k, dy, dx, 16.0)
        p["Uh"][out, 16 + k] = 2048.0
        p["bh"][out] = -2048.0

    # -----------------------------------------------------------
    # Global readout.
    # -----------------------------------------------------------

    for k in range(4):
        p["Wz"][k, 12 + k] = 1.0
    p["Wz"][8, 32 + 3] = 1.0
    p["Wz"][10, 64 + 10] = 1.0
    p["Wz"][11, 64 + 11] = 1.0

    # Unique, size-compatible target coordinate preferences.
    # For small targets an explicit count disables this preference
    # when several targets remain; the navigation field then chooses.
    for kind, (width, height) in enumerate(sizes):
        for d, (dy, dx) in enumerate(directions):
            unit = 32 + 4 * kind + d
            axis = 0 if dy else 1
            sign = dy if dy else dx

            p["Wz"][unit, 32 + 20 + 2 * kind + axis] = 63.0 * sign
            p["Wz"][unit, 32 + 14 + axis] = -63.0 * sign
            p["Wz"][unit, 32 + 28] = (
                200.0 if width == 6 else -200.0
            )
            p["Wz"][unit, 32 + 29] = (
                200.0 if height == 6 else -200.0
            )

            ones = int(width == 6) + int(height == 6)
            adjustment = int(
                (height == 6) if axis == 0 else (width == 6)
            )
            p["bz"][unit] = -200.0 * ones + sign * adjustment

            if kind == 0:
                p["Wz"][unit, 32 + 30] = 256.0
                p["Wz"][unit, 64 + 30] = -128.0 * 4096.0
                p["bz"][unit] -= 128.0
            else:
                # A missing target contributes no coordinate preference.
                # Its coordinate features are zero in that case.
                p["bz"][unit] -= 64.0
                for a in (0, 1):
                    p["Wz"][unit, 32 + 20 + 2 * kind + a] += 64.0

    # -----------------------------------------------------------
    # Memory: geometry-change tests and reachable-reading flags.
    # -----------------------------------------------------------

    scale = 0.005
    x_weight = math.sqrt(2.0)
    gain = 1.0e7
    p["Wmz"][0, 10] = scale
    p["Wmz"][0, 11] = scale * x_weight

    for k, action in enumerate((1, 2, 3, 4)):
        for j, sign in enumerate((1.0, -1.0)):
            unit = 1 + 2 * k + j
            p["Wm"][unit, 0] = -sign * gain
            p["Wmz"][unit, 10] = sign * gain * scale
            p["Wmz"][unit, 11] = sign * gain * scale * x_weight
            p["Wma"][unit, action] = 50.0
            p["bm"][unit] = -49.0

        p["Wma"][9 + k, action] = 50.0
        p["bm"][9 + k] = -25.0

        p["Wma"][16 + k, action] = 10.0
        p["bm"][16 + k] = -5.0

        # A zero directional reading must not be rescued by a global
        # coordinate preference across an impassable gulch.
        p["Wmz"][24 + k, k] = 100.0
        p["bm"][24 + k] = -10.0

    p["Wma"][14, :] = 50.0
    p["bm"][14] = -25.0

    # -----------------------------------------------------------
    # Action and click heads.
    # -----------------------------------------------------------

    p["bo"][:] = -1000.0
    opposite = [1, 0, 3, 2]

    for k, action in enumerate((1, 2, 3, 4)):
        p["bo"][action] = -500.0
        p["Wo"][action, k] = 1.0

        p["Wo"][action, 128 + 1 + 2 * k] = -100.0
        p["Wo"][action, 128 + 2 + 2 * k] = -100.0
        p["Wo"][action, 128 + 9 + k] = 100.0
        p["Wo"][action, 128 + 14] = -100.0

        p["Wo"][action, 128 + 16 + opposite[k]] = -0.1
        p["Wo"][action, 128 + 24 + k] = 500.0

        p["Wo"][action, 32 + k] = 2.0
        for kind in range(1, 4):
            p["Wo"][action, 32 + 4 * kind + k] = 0.15

    # Select another piece when no useful movement remains.
    p["bo"][6] = 0.3
    p["Wc"][0, 9] = 100.0
    p["Wc"][0, 13] = 1.0
    p["Wc"][0, 2] = -1000.0
    p["Wc"][0, 10] = -0.0001
    p["Wc"][0, 11] = -0.00007

    return p
