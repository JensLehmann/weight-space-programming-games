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

    def c1(out, inp, value=1.0, dy=0, dx=0):
        p["W1"][out, inp, dy + 2, dx + 2] += value

    def c2(out, inp, value=1.0, dy=0, dx=0):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    def c3(out, inp, value=1.0):
        p["W3"][out, inp] += value

    def ch(out, inp, value=1.0, dy=0, dx=0):
        p["Wh"][out, inp, dy + 1, dx + 1] += value

    cardinal = [
        (-1, 0),  # north
        (1, 0),   # south
        (0, -1),  # west
        (0, 1),   # east
    ]
    # Present arms, absent arms, and their intervening diagonal.
    corners = [
        (0, 3, 1, 2, -1, 1),
        (1, 3, 0, 2, 1, 1),
        (1, 2, 0, 3, 1, -1),
        (0, 2, 1, 3, -1, -1),
    ]

    # ------------------------------------------------------------
    # First layer: colours, coordinates, and local geometry.
    # ------------------------------------------------------------
    for out, colour in ((0, 0), (1, 11), (2, 4), (3, 5)):
        c1(out, colour)

    # A selected mirror has a black centre surrounded by colour 10.
    for out, vertical in ((4, True), (5, False)):
        c1(out, 0, 4)
        for offset in (-2, -1, 1, 2):
            c1(
                out, 10, 1,
                offset if vertical else 0,
                0 if vertical else offset,
            )
        for offset in (-2, 2):
            c1(
                out, 10, -1,
                0 if vertical else offset,
                offset if vertical else 0,
            )
        p["b1"][out] = -7

    c1(6, 17)
    c1(7, 16)

    # Target-gated coordinates and complementary coordinates.
    for out, coordinate, increasing in (
        (8, 17, True),
        (9, 17, False),
        (10, 16, True),
        (11, 16, False),
    ):
        c1(out, 11)
        c1(out, coordinate, 1 if increasing else -1)
        p["b1"][out] = -1 if increasing else 0

    # Exactly one on the energy-bar column.
    c1(12, 17, 63)
    p["b1"][12] = -62

    # Occupancy sampled two pixels from the cell centre.
    for j, (dy, dx) in enumerate(cardinal):
        c1(13 + j, 5, 1, 2 * dy, 2 * dx)
        c1(13 + j, 4, 1, 2 * dy, 2 * dx)

    for j, (_, _, _, _, dy, dx) in enumerate(corners):
        c1(17 + j, 5, 1, 2 * dy, 2 * dx)
        c1(17 + j, 4, 1, 2 * dy, 2 * dx)

    # Target samples, extended to three pixels by the next layer.
    for j, (dy, dx) in enumerate(cardinal):
        c1(21 + j, 11, 1, 2 * dy, 2 * dx)

    for j, (_, _, _, _, dy, dx) in enumerate(corners):
        c1(25 + j, 11, 1, 2 * dy, 2 * dx)

    # Physical cell centre, allowing a covered target centre.
    c1(29, 0, 4)
    c1(29, 11, 4)
    for dy, dx in cardinal:
        c1(29, 5, 1, dy, dx)
    p["b1"][29] = -7

    # A covered reflected cell has an 11-coloured centre.
    c1(30, 11, 4)
    for dy, dx in cardinal:
        c1(30, 4, 1, dy, dx)
    p["b1"][30] = -7

    # Unambiguously selected physical centre.
    c1(31, 0, 4)
    for dy, dx in cardinal:
        c1(31, 5, 1, dy, dx)
    p["b1"][31] = -7

    # ------------------------------------------------------------
    # Second layer: physical/ghost maps and corner descriptors.
    # ------------------------------------------------------------
    c2(0, 29)
    c2(1, 2)
    c2(1, 30)

    c2(2, 1)
    c2(2, 12, -1)

    c2(3, 4)
    c2(4, 5)
    c2(5, 6)
    c2(6, 7)

    for out, inp in zip(range(7, 11), range(8, 12)):
        c2(out, inp)
        c2(out, 12, -1)

    c2(11, 31)

    # Source open/filled corners in all four orientations.
    for j, (a, b, absent_a, absent_b, dy, dx) in enumerate(corners):
        for filled, out in ((False, 12 + j), (True, 16 + j)):
            c2(out, 29)
            c2(out, 2)
            c2(out, 30)
            c2(out, 13 + a)
            c2(out, 13 + b)
            c2(out, 13 + absent_a, -1)
            c2(out, 13 + absent_b, -1)
            c2(out, 17 + j, 1 if filled else -1)
            p["b2"][out] = -3 if filled else -2

        # Target descriptors use neighbouring cell centres exactly.
        for filled, out in ((False, 20 + j), (True, 24 + j)):
            c2(out, 1)
            c2(out, 12, -1)

            for neighbour, coefficient in (
                (a, 1),
                (b, 1),
                (absent_a, -1),
                (absent_b, -1),
            ):
                sy, sx = cardinal[neighbour]
                c2(out, 21 + neighbour, coefficient, sy, sx)

            c2(out, 25 + j, 1 if filled else -1, dy, dx)
            p["b2"][out] = -3 if filled else -2

    # ------------------------------------------------------------
    # Third layer: clipped reflection-aware masks.
    #
    # Channels 0..7 hold raw masks; 8..15 hold excess above one.
    # Their differences therefore lie in [0,1].
    # ------------------------------------------------------------
    for kind, source_channels in enumerate(
        (range(12, 16), range(16, 20))
    ):
        for direction in range(4):
            raw = 4 * kind + direction
            excess = 8 + raw

            for out in (raw, excess):
                for inp in source_channels:
                    c3(out, inp)

                if direction == 0:
                    # Normal X: actual source or horizontal image.
                    c3(out, 64 + 11)
                    bias = -1

                elif direction == 1:
                    # Reversed X: a vertical image.
                    c3(out, 1)
                    c3(out, 32 + 11)
                    c3(out, 64 + 11, -1)
                    bias = -2

                elif direction == 2:
                    # Normal Y: actual source or vertical image.
                    c3(out, 32 + 11)
                    bias = -1

                else:
                    # Reversed Y: a horizontal image.
                    c3(out, 1)
                    c3(out, 64 + 11)
                    c3(out, 32 + 11, -1)
                    bias = -2

                p["b3"][out] = bias - int(out == excess)

    for inp in range(20, 24):
        c3(16, inp)
    for inp in range(24, 28):
        c3(17, inp)

    c3(18, 3)
    c3(19, 4)

    c3(20, 5)
    c3(20, 3)
    p["b3"][20] = -1

    c3(21, 6)
    c3(21, 4)
    p["b3"][21] = -1

    for out, inp in zip(range(22, 26), range(7, 11)):
        c3(out, inp)

    # Physical-corner class flags. Projection onto selected centres
    # excludes already-covered, unselected pieces where possible.
    for out, source_channels in (
        (26, range(12, 16)),
        (27, range(16, 20)),
    ):
        for inp in source_channels:
            c3(out, inp, 4)
        c3(out, 0, 4)
        c3(out, 32 + 11)
        c3(out, 64 + 11)
        p["b3"][out] = -8

    c3(28, 11)
    c3(29, 2)

    # ------------------------------------------------------------
    # Recurrent block: two target potentials and signed derivatives.
    # ------------------------------------------------------------
    field_strength = 128.0
    baseline = 512.0
    exclusion = 2048.0

    terms = (
        ((0, 1, 1), (0, -1, -1)),   # normal X
        ((0, -1, 1), (0, 1, -1)),   # reversed X
        ((1, 0, 1), (-1, 0, -1)),   # normal Y
        ((-1, 0, 1), (1, 0, -1)),   # reversed Y
    )

    for kind in range(2):
        p["Uh"][kind, 16 + kind] = field_strength

        # Finite-horizon, critically weighted diffusion.
        for dy, dx in cardinal:
            ch(kind, kind, 0.25, dy, dx)

        for direction in range(4):
            out = 2 + 4 * kind + direction
            mask = 4 * kind + direction

            for dy, dx, coefficient in terms[direction]:
                ch(out, kind, coefficient, dy, dx)

            p["Uh"][out, mask] = exclusion + baseline
            p["Uh"][out, 8 + mask] = -(exclusion + baseline)
            p["bh"][out] = -exclusion

    def hmean(channel):
        return 16 + channel

    def fmax(channel):
        return 32 + channel

    def fmean(channel):
        return 64 + channel

    # ------------------------------------------------------------
    # Global force readout. The constant mask baseline is removed.
    # Each corner class has right/left/down/up positive force units.
    # ------------------------------------------------------------
    force_scale = 4096.0 / field_strength

    for kind in range(2):
        other_class = 27 if kind == 0 else 26

        for local_unit, axis, sign in (
            (0, 0, 1),
            (1, 0, -1),
            (2, 1, 1),
            (3, 1, -1),
        ):
            unit = 4 * kind + local_unit
            selected_directions = (0, 1) if axis == 0 else (2, 3)

            for direction in selected_directions:
                mask = 4 * kind + direction
                p["Wz"][unit, hmean(2 + mask)] += (
                    sign * force_scale
                )
                p["Wz"][unit, fmean(mask)] -= (
                    sign * force_scale * baseline
                )
                p["Wz"][unit, fmean(8 + mask)] += (
                    sign * force_scale * baseline
                )

            # Suppress the wrong corner class without introducing a
            # large cancelling constant into the relevant force.
            p["Wz"][unit, fmax(other_class)] = -100000
            p["Wz"][unit, fmax(18)] = -100000
            p["Wz"][unit, fmax(19)] = -100000

    # Selected mirrors are aligned to target bounding-box midpoints.
    axis_scale = 16384.0
    enable = 32768.0
    tolerance = 1.45 / 63.0

    for positive, negative, hi, lo, position, flag in (
        (8, 9, 22, 23, 20, 18),
        (10, 11, 24, 25, 21, 19),
    ):
        for unit, sign in ((positive, 1), (negative, -1)):
            p["Wz"][unit, fmax(hi)] = sign * axis_scale * 0.5
            p["Wz"][unit, fmax(lo)] = -sign * axis_scale * 0.5
            p["Wz"][unit, fmax(position)] = -sign * axis_scale
            p["Wz"][unit, fmax(flag)] = enable
            p["bz"][unit] = (
                axis_scale * (sign * 0.5 - tolerance) - enable
            )

    p["Wz"][12, fmax(18)] = 1
    p["Wz"][13, fmax(19)] = 1

    # ------------------------------------------------------------
    # Memory: resettable selection timeout and previous direction.
    # ------------------------------------------------------------
    p["bm"][63] = -20

    p["Wm"][0, 63] = 40
    p["Wma"][0, 5] = 40
    p["bm"][0] = 20

    for unit in range(1, 24):
        p["Wm"][unit, unit - 1] = 40
        p["Wma"][unit, 5] = -80
        p["bm"][unit] = -20

    p["Wm"][24, 23] = 40
    p["Wm"][24, 63] = -40

    for unit, action in zip(range(30, 34), (1, 2, 3, 4)):
        p["Wma"][unit, action] = 40
        p["bm"][unit] = -20

    # ------------------------------------------------------------
    # Actions: sum relevant class forces and mirror corrections.
    # ------------------------------------------------------------
    reversal_penalty = 0.05

    for action, force, mirror, opposite_flag in (
        (1, 3, 11, 31),  # up
        (2, 2, 10, 30),  # down
        (3, 1, 9, 33),   # left
        (4, 0, 8, 32),   # right
    ):
        p["Wo"][action, force] = 1
        p["Wo"][action, 4 + force] = 1
        p["Wo"][action, mirror] = 1
        p["Wo"][action, 128 + opposite_flag] = (
            -0.5 * reversal_penalty
        )
        p["bo"][action] = -0.5 * reversal_penalty

    p["Wo"][5, 12] = 0.001
    p["Wo"][5, 13] = 0.001
    p["Wo"][5, 128 + 24] = 10000
    p["bo"][5] = 1e-6

    p["bo"][0] = -10000
    p["bo"][6] = -10000
    p["bo"][7] = -10000

    return p
