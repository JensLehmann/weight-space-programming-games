import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,),
        "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,),
        "Wh": (16, 16, 3, 3), "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,),
        "Wm": (64, 64), "Wmz": (64, 128),
        "Wma": (64, 8), "bm": (64,),
        "Wo": (8, 192), "bo": (8,),
        "Wc": (1, 48), "bc": (1,),
    }
    p = {name: np.zeros(shape, dtype=np.float32)
         for name, shape in shapes.items()}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    sizes = [(3, 3), (3, 6), (6, 3), (6, 6)]

    def put1(out, colors, offset=(0, 0), weight=1.0):
        y, x = offset
        for color in colors:
            p["W1"][out, color, y + 2, x + 2] += weight

    def put2(out, channel, offset=(0, 0), weight=1.0):
        y, x = offset
        p["W2"][out, channel, y + 1, x + 1] += weight

    def puth(out, channel, offset=(0, 0), weight=1.0):
        y, x = offset
        p["Wh"][out, channel, y + 1, x + 1] += weight

    def template(out, requirements, forbidden=()):
        for offset, colors in requirements:
            put1(out, colors, offset)
        for offset, colors in forbidden:
            put1(out, colors, offset, -1.0)
        p["b1"][out] = 1.0 - len(requirements)

    # First layer: terrain, goal, piece, and pushing-position templates.
    put1(0, [2])
    put1(1, [15])
    put1(2, [4], (0, -2))

    req = [((y, x), [4]) for y in (-2, 2) for x in (-2, 2)]
    req += [(at, [1])
            for at in [(-2, 0), (2, 0), (0, -2), (0, 2)]]
    template(3, req)

    corner = [
        ((-2, -2), [4]), ((-2, -1), [4]), ((-2, 0), [4]),
        ((-1, -2), [4]), ((0, -2), [4]),
    ]
    for i, (width, height) in enumerate(sizes):
        if width == height == 3:
            req = [
                ((y, x), [4])
                for y in range(-2, 3)
                for x in range(-2, 3)
                if abs(y) == 2 or abs(x) == 2
            ]
            forbidden = []
        else:
            req = list(corner)
            forbidden = []
            if width == 3:
                req.append(((-1, 2), [4]))
            else:
                req.append(((-2, 2), [4]))
                forbidden.append(((-1, 2), [4]))
            if height == 3:
                req.append(((2, -1), [4]))
            else:
                req.append(((2, -2), [4]))
                forbidden.append(((2, -1), [4]))
        template(4 + i, req, forbidden)

    border = [0, 14]

    def piece_template(width, height, selected):
        inside = [0] if selected else [0, 4, 5]

        if width == height == 3:
            req = [((0, 0), inside)]
            req += [
                ((y, x), border)
                for y in (-1, 0, 1)
                for x in (-1, 0, 1)
                if y or x
            ]
            return req

        if width == 3 and height == 6:
            req = [
                ((0, 0), [14]), ((1, 0), inside), ((2, 0), inside)
            ]
            req += [((-1, x), border) for x in (-1, 0, 1)]
            for y in (0, 1, 2):
                req += [((y, -1), border), ((y, 1), border)]
            return req

        if width == 6 and height == 3:
            return [
                ((x, y), colors)
                for (y, x), colors
                in piece_template(3, 6, selected)
            ]

        req = [
            ((1, 1), inside), ((1, 2), inside),
            ((2, 1), inside), ((2, 2), inside),
        ]
        req += [((-1, x), border) for x in (-1, 0, 1, 2)]
        req += [((y, -1), border) for y in (0, 1, 2)]
        req += [
            (at, [14])
            for at in [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)]
        ]
        return req

    for i, size in enumerate(sizes):
        template(8 + i, piece_template(*size, False))
        template(12 + i, piece_template(*size, True))

    for i, (dy, dx) in enumerate(directions):
        py, px = -dx, dy
        req = [
            ((2 * dy + t * py, 2 * dx + t * px),
             [14, 11, 12, 13])
            for t in (-1, 0, 1)
        ]
        template(16 + i, req)
        put1(20 + i, [4, 5, 11, 12, 13], (2 * dy, 2 * dx))

    put1(24, [14])
    put1(25, [11, 12, 13])

    p["W1"][26, 16, 2, 2] = 1.0
    p["b1"][26] = 1.0
    p["W1"][27, 17, 2, 2] = 1.0
    p["b1"][27] = 1.0

    for i, (dy, dx) in enumerate(directions):
        py, px = -dx, dy
        for t in (-1, 0, 1):
            put1(28 + i, [2, 15],
                 (2 * dy + t * py, 2 * dx + t * px))

    # Second layer: dilated terrain and goal occupancy.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            put2(0, 0, (dy, dx))
            put2(1, 0, (dy, dx))
            put2(1, 1, (dy, dx))

    put2(2, 27)
    put2(30, 26)
    put2(7, 3)

    for i in range(4):
        put2(3 + i, 4 + i)
        put2(3 + i, 8 + i, weight=-1.0)
        put2(3 + i, 2, (1, -1), -1.0)

        put2(8 + i, 12 + i)
        put2(12, 8 + i)
        put2(13, 8 + i)
        put2(13, 12 + i, weight=-1.0)
        put2(31, 12 + i)

        put2(22 + i, 4 + i)
        put2(22 + i, 8 + i)
        put2(22 + i, 2, (1, -1), -1.0)
        p["b2"][22 + i] = -1.0

    p["b2"][31] = -1.0

    for i, direction in enumerate(directions):
        put2(18 + i, 16 + i)
        put2(18 + i, 20 + i, direction)
        put2(18 + i, 24, weight=-3.0)
        put2(18 + i, 25, weight=-3.0)
        p["b2"][18 + i] = -1.0

        put2(26 + i, 28 + i)
        put2(26 + i, 28 + i, direction)
        for j in range(4):
            put2(26 + i, 12 + j, weight=100.0)
        put2(26 + i, 31, weight=-100.0)
        p["b2"][26 + i] = -100.0

    # Third layer: compact semantic planes.
    p["W3"][0, 0] = 1.0
    p["W3"][1, 1] = 1.0
    p["W3"][3, 3:7] = 1.0
    p["W3"][4, 7] = 1.0

    p["W3"][5, 8:12] = 1.0
    p["W3"][5, 31] = -1.0

    p["W3"][12, 13] = 1.0
    p["W3"][13, 22:26] = 1.0

    p["W3"][14, 22:26] = 1.0
    p["W3"][14, 8:12] = 1.0
    p["W3"][14, 31] = -1.0
    p["b3"][14] = -1.0

    for i in range(4):
        p["W3"][8 + i, 18 + i] = 1.0
        p["W3"][16 + i, 26 + i] = 1.0

    for out, coord in [(30, 30), (31, 2)]:
        p["W3"][out, coord] = 1.0
        p["W3"][out, 8:12] = 100.0
        p["W3"][out, 31] = -100.0
        p["b3"][out] = -100.0

    # Stable diffusion: the kernel sums to one. Obstacles clamp the
    # corresponding potential to zero through a large negative drive.
    def diffusion(channel, obstacle):
        puth(channel, channel, weight=0.04)
        for offset in directions:
            puth(channel, channel, offset, 0.24)
        p["Uh"][channel, obstacle] = -100000.0

    # Transport potentials allow gulch crossings.
    diffusion(0, 0)
    diffusion(1, 0)
    p["Uh"][0, 3] = 1.0
    p["Uh"][1, 4] = 1.0

    # Direct walking potential excludes gulches.
    diffusion(2, 1)
    p["Uh"][2, 3] = 1.0

    # Four directional pushing-position maps. Rectangular goals take
    # precedence while present; the cross potential supplies a fallback.
    for i, offset in enumerate(directions):
        out = 4 + i
        puth(out, 0, offset)
        puth(out, 0, weight=-1.0)
        puth(out, 1, offset, 0.1)
        puth(out, 1, weight=-0.1)
        p["Uh"][out, 8 + i] = 16.0
        p["Uh"][out, 1] = -16.0
        p["bh"][out] = -16.0

    # Walking potential toward useful pushing positions.
    diffusion(3, 1)
    for stance in range(4, 8):
        puth(3, stance, weight=0.2)

    # Selected-anchor direction scores for direct and helper navigation.
    for i, offset in enumerate(directions):
        direct = 8 + i
        puth(direct, 2, offset)
        puth(direct, 2, weight=-1.0)
        p["Uh"][direct, 5] = 32.0
        p["bh"][direct] = -32.0

        helper = 12 + i
        puth(helper, 3, offset)
        puth(helper, 3, weight=-1.0)
        puth(helper, 4 + i, weight=20.0)
        p["Uh"][helper, 5] = 64.0
        p["bh"][helper] = -64.0

    # Paired readout rails clip only exceptionally large evidence.
    for i in range(8):
        p["Wz"][i, 8 + i] = 1.0
        p["Wz"][8 + i, 8 + i] = 1.0
        p["bz"][8 + i] = -10.0

    p["Wz"][16, 32 + 14] = 1.0
    p["Wz"][17, 32 + 30] = 1.0
    p["Wz"][18, 32 + 31] = 1.0
    for i in range(4):
        p["Wz"][20 + i, 32 + 16 + i] = 1.0

    # Helper-mode latch: placement enables it; selection of an unfinished
    # piece disables it. A placed piece remains usable as a helper.
    p["Wm"][10, 10] = 5.0
    p["Wmz"][10, 16] = 20.0
    p["Wma"][10, 6] = -15.0
    p["bm"][10] = -1.0
    p["bm"][11] = 10.0

    # Preserve relative movement strengths instead of saturating every
    # positive direction to essentially the same value.
    for i in range(4):
        p["Wmz"][i, i] = 0.5
        p["Wmz"][i, 8 + i] = -0.5
        p["Wm"][i, 10] = -10.0
        p["Wm"][i, 11] = -10.0

        p["Wmz"][4 + i, 4 + i] = 0.5
        p["Wmz"][4 + i, 12 + i] = -0.5
        p["Wm"][4 + i, 10] = 10.0
        p["Wm"][4 + i, 11] = 10.0
        p["bm"][4 + i] = -20.0

    # Placement requests reselection, except immediately after a click.
    p["Wmz"][13, 16] = 5.0
    p["Wma"][13, 6] = -10.0

    # Store the selected anchor's coordinates at a nearly linear scale.
    p["Wmz"][40, 17] = 0.01
    p["Wmz"][41, 18] = 0.01

    # Movement-success comparators. Strong previous-action gating makes
    # them inactive at episode start and following a selection change.
    for i, action in enumerate((1, 2, 3, 4)):
        coord_z = 17 if i < 2 else 18
        coord_m = 40 if i < 2 else 41
        sign = -1.0 if i in (0, 2) else 1.0
        comparison = 44 + i
        previous = 52 + i

        p["Wmz"][comparison, coord_z] = sign * 100.0
        p["Wm"][comparison, coord_m] = -sign * 10000.0
        p["Wma"][comparison, action] = 1000.0
        p["bm"][comparison] = -1001.0

        p["Wma"][previous, action] = 20.0
        p["bm"][previous] = -10.0

    for i, action in enumerate((1, 2, 3, 4)):
        p["Wo"][action, 128 + i] = 1000.0
        p["Wo"][action, 128 + 4 + i] = 1000.0
        p["Wo"][action, 20 + i] = -10000.0

        p["Wo"][action, 128 + 44 + i] = 10.0
        p["Wo"][action, 128 + 52 + i] = -10.0

        # A push moves the target while leaving the selected helper
        # stationary, so it is a useful occasion to reselect.
        p["Wo"][6, 128 + 52 + i] = 80.0
        p["Wo"][6, 128 + 44 + i] = -80.0

    p["Wo"][6, 128 + 13] = 3000.0
    p["bo"][6] = -999.9999

    p["bo"][0] = -100000.0
    p["bo"][5] = -100000.0
    p["bo"][7] = -100000.0

    # Click only nonselected piece anchors. Unfinished pieces dominate
    # placed helpers; transport potential breaks ties spatially.
    p["Wc"][0, 12] = 10000.0
    p["Wc"][0, 13] = -1000.0
    p["Wc"][0, 5] = -20000.0
    p["Wc"][0, 32] = 100.0
    p["Wc"][0, 33] = 10.0

    return p
