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

    colours = (8, 9, 11)
    ink_colours = (0, 6, 7, 8, 9, 10, 11, 12, 13, 14)
    neighbours = [
        (dy, dx)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if dy != 0 or dx != 0
    ]

    def first(out, inp, value=1.0, dy=0, dx=0):
        p["W1"][out, inp, 2 + dy, 2 + dx] += value

    def second(out, inp, value=1.0):
        p["W2"][out, inp, 1, 1] += value

    def third(out, inp, value=1.0, projection=0):
        p["W3"][out, inp + 32 * projection] += value

    # First layer: marker, drawable ink, glyph-centre hinges.
    first(0, 0)
    for colour in ink_colours:
        first(1, colour)

    for dy, dx in neighbours:
        first(2, 4, dy=dy, dx=dx)
        first(3, 4, dy=dy, dx=dx)
    p["b1"][2] = -3.0
    p["b1"][3] = -4.0

    for i, colour in enumerate(colours):
        first(4 + i, colour)
        first(4 + i, 0)

    first(7, 16)
    first(8, 17)

    # Dense patches identify palette-well interiors.
    for i, colour in enumerate(colours):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                first(9 + i, colour, dy=dy, dx=dx)
        p["b1"][9 + i] = -8.0

    # Hinge pairs for colour adjacent to the selected marker.
    for i, colour in enumerate(colours):
        a = 12 + 2 * i
        for out in (a, a + 1):
            first(out, 0, 16.0)
            for dy, dx in neighbours:
                first(out, colour, dy=dy, dx=dx)
        p["b1"][a] = -16.0
        p["b1"][a + 1] = -17.0

    for out, offsets in (
        (18, ((0, -1), (0, 1))),
        (19, ((-1, 0), (1, 0))),
        (20, ((-1, -1), (1, 1))),
    ):
        first(out, 0, 3.0)
        for dy, dx in offsets:
            for colour in ink_colours:
                first(out, colour, dy=dy, dx=dx)
        p["b1"][out] = -4.0

    first(21, 0)
    for dy, dx in neighbours:
        for colour in ink_colours:
            first(21, colour, -1.0, dy, dx)

    # Second layer.
    second(0, 1)
    for i in range(3):
        second(0, 9 + i, -1.0)
    second(1, 0)
    second(2, 21)

    for i in range(3):
        # Coloured outline, excluding marker, glyph centre, and well.
        second(3 + i, 4 + i)
        second(3 + i, 0, -1.0)
        second(3 + i, 2, -1.0)
        second(3 + i, 3)
        second(3 + i, 9 + i, -1.0)

        # Binary adjacent-marker colour.
        second(6 + i, 12 + 2 * i)
        second(6 + i, 13 + 2 * i, -1.0)

        # Four target-coordinate carriers: x, 1-x, y, 1-y.
        # Adding the first two later recovers the binary target mask.
        base = 9 + 4 * i
        for offset, coord, sign in (
            (0, 8, 1.0),
            (1, 8, -1.0),
            (2, 7, 1.0),
            (3, 7, -1.0),
        ):
            out = base + offset
            second(out, coord, sign)
            second(out, 4 + i)
            second(out, 2)
            second(out, 3, -1.0)
            p["b2"][out] = -2.0 if sign > 0 else -1.0

        # Palette-well coordinates.
        for axis, coord in enumerate((8, 7)):
            out = 21 + 2 * i + axis
            second(out, coord)
            second(out, 9 + i)
            p["b2"][out] = -1.0

    second(27, 8)
    second(27, 0)
    p["b2"][27] = -1.0
    second(28, 7)
    second(28, 0)
    p["b2"][28] = -1.0

    second(29, 18)
    second(29, 19, -1.0)
    second(30, 19)
    second(30, 18, -1.0)
    second(31, 20)

    # Axial layer.
    third(0, 0)
    third(1, 1)

    # Hollow-outline seeds at intersections with its marker row.
    third(2, 0)
    third(2, 2, projection=1)
    p["b3"][2] = -1.0

    for i in range(3):
        base = 9 + 4 * i
        third(3 + i, base)
        third(3 + i, base + 1)

        # Selected colour: local adjacent-marker colour, or hollow
        # outline colour on the isolated marker's row.
        third(6 + i, 6 + i, 2.0)
        third(6 + i, 3 + i)
        third(6 + i, 2, projection=1)
        p["b3"][6 + i] = -1.0

        third(9 + 2 * i, 21 + 2 * i)
        third(10 + 2 * i, 22 + 2 * i)

    third(15, 27)
    third(16, 28)
    third(17, 29)
    third(18, 30)
    third(19, 31)

    # A paired small coordinate perturbation identifies the position
    # of the widest target row / tallest target column after max pooling.
    eps = 0.01
    for i in range(3):
        base = 9 + 4 * i
        out = 20 + 4 * i
        for sign, channel in ((1.0, out), (-1.0, out + 1)):
            third(channel, base, projection=1)
            third(channel, base + 1, projection=1)
            third(channel, base + 2, sign * eps, projection=1)
            p["b3"][channel] = -0.5
        for sign, channel in ((1.0, out + 2), (-1.0, out + 3)):
            third(channel, base + 2, projection=2)
            third(channel, base + 3, projection=2)
            third(channel, base, sign * eps, projection=2)
            p["b3"][channel] = -0.5

    # Tied block: binary marker-connected ink.
    for out in (0, 1):
        p["Wh"][out, 0, :, :] = 1.0
        p["Wh"][out, 1, :, :] = -1.0
        p["Uh"][out, 0] = 16.0
        p["Uh"][out, 1] = 1.0
        p["Uh"][out, 2] = 1.0
    p["bh"][0] = -16.0
    p["bh"][1] = -17.0

    # Bounded harmonic potential, h[2]-h[3].
    for out in (2, 3):
        p["Wh"][out, 0, 1, 1] = 4.0
        p["Wh"][out, 1, 1, 1] = -4.0
        for dy, dx in neighbours:
            weight = 0.20 if dy == 0 or dx == 0 else 0.0485
            p["Wh"][out, 2, 1 + dy, 1 + dx] = weight
            p["Wh"][out, 3, 1 + dy, 1 + dx] = -weight
    p["bh"][3] = -1.0

    # Colour-specific directional coverage changes.
    probes = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for i in range(3):
        for d, (dy, dx) in enumerate(probes):
            out = 4 + 4 * i + d
            p["Wh"][out, 2, 1 + dy, 1 + dx] += 1.0
            p["Wh"][out, 3, 1 + dy, 1 + dx] -= 1.0
            p["Wh"][out, 2, 1, 1] -= 1.0
            p["Wh"][out, 3, 1, 1] += 1.0
            p["Uh"][out, 3 + i] = 4.0
            p["bh"][out] = -3.0

    def fmax(c):
        return 32 + c

    def fmean(c):
        return 64 + c

    def hmean(c):
        return 16 + c

    count = 0

    def unit(coefficients, bias=0.0):
        nonlocal count
        if count >= 128:
            raise RuntimeError("Internal unit allocation exceeded 128")
        out = count
        count += 1
        for inp, value in coefficients.items():
            p["Wz"][out, inp] += value
        p["bz"][out] = bias
        return out

    def add(coefficients, inp, value):
        coefficients[inp] = coefficients.get(inp, 0.0) + value

    votes = {}
    for i in range(3):
        for d in range(4):
            votes[i, d] = unit({
                hmean(4 + 4 * i + d): 4096.0,
                fmean(3 + i): -4096.0,
            }, -0.0001)

    # Gradient control is enabled for matching diagonal/hollow pieces.
    inactive = []
    for i in range(3):
        inactive.append(unit({
            fmax(6 + i): -1.0,
            fmax(19): -1.0,
            fmax(2): -1.0,
        }, 2.0))

    for i in range(3):
        baseline = 13 + i
        p["Wmz"][baseline, inactive[i]] = -12.0
        for d in range(4):
            memory_unit = 1 + 4 * i + d
            p["Wmz"][memory_unit, votes[i, d]] = 2.0
            p["Wmz"][memory_unit, inactive[i]] = -12.0
            p["Wo"][1 + d, 128 + memory_unit] += 1.0
            p["Wo"][1 + d, 128 + baseline] -= 1.0

    # Direct cross/bar positioning. A 1.4-pixel dead band accounts
    # for the three-pixel action lattice.
    coordinate_gain = 63.0 / (2.0 * eps)
    for i in range(3):
        span = 20 + 4 * i
        controls = (
            {
                fmax(16): 63.0,
                fmax(span): -coordinate_gain,
                fmax(span + 1): coordinate_gain,
            },
            {
                fmax(16): -63.0,
                fmax(span): coordinate_gain,
                fmax(span + 1): -coordinate_gain,
            },
            {
                fmax(15): 63.0,
                fmax(span + 2): -coordinate_gain,
                fmax(span + 3): coordinate_gain,
            },
            {
                fmax(15): -63.0,
                fmax(span + 2): coordinate_gain,
                fmax(span + 3): -coordinate_gain,
            },
        )
        for d, control in enumerate(controls):
            coefficients = dict(control)
            add(coefficients, fmax(6 + i), 128.0)
            add(coefficients, fmax(3 + i), 128.0)
            add(coefficients, fmax(19), -128.0)
            add(coefficients, fmax(2), -128.0)
            u = unit(coefficients, -256.0 - 1.4)
            p["Wo"][1 + d, u] = 0.10

    # Global recolouring-need predicates.
    need = [
        unit({
            fmax(6): -1.0,
            fmax(7): -1.0,
            fmax(8): -1.0,
        }, 1.0)
    ]
    for i in range(3):
        need.append(unit({
            fmax(6 + i): 1.0,
            fmax(3 + i): -1.0,
        }))

    # Colour preference depends on visible shape family and remaining
    # targets, not on a level number.
    families = (
        (0.0, {fmax(2): 1.0}, (0, 1, 2)),
        (0.0, {fmax(19): 1.0}, (1, 0, 2)),
        (1.0, {
            fmax(19): -1.0,
            fmax(2): -1.0,
        }, (1, 2, 0)),
    )

    routing = [[] for _ in range(4)]
    for family_bias, family, order in families:
        earlier = []
        for i in order:
            chosen = {fmax(3 + i): 1.0}
            for j in earlier:
                chosen[fmax(3 + j)] = -1.0
            earlier.append(i)

            well_x = fmax(9 + 2 * i)
            well_y = fmax(10 + 2 * i)
            controls = (
                {fmax(16): 63.0, well_y: -63.0},
                {fmax(16): -63.0, well_y: 63.0},
                {fmax(15): 63.0, well_x: -63.0},
                {fmax(15): -63.0, well_x: 63.0},
            )
            for d, control in enumerate(controls):
                coefficients = dict(control)
                for condition in (family, chosen):
                    for inp, value in condition.items():
                        add(coefficients, inp, 128.0 * value)
                u = unit(
                    coefficients,
                    128.0 * (family_bias - 2.0) - 0.8,
                )
                routing[d].append(u)

    # Memory gates the shared routing scores. The baseline subtraction
    # removes inactive-route saturation.
    route_baseline = 20
    p["bm"][route_baseline] = -16.0
    for u in need:
        p["Wmz"][route_baseline, u] = 16.0

    for d in range(4):
        memory_unit = 16 + d
        p["bm"][memory_unit] = -16.0
        for u in need:
            p["Wmz"][memory_unit, u] = 16.0
        for u in routing[d]:
            p["Wmz"][memory_unit, u] = 0.10
        p["Wo"][1 + d, 128 + memory_unit] += 2.0
        p["Wo"][1 + d, 128 + route_baseline] -= 2.0

    p["bo"][0] = -1000.0
    p["bo"][5] = 0.0001
    p["bo"][6] = -1000.0
    p["bo"][7] = -1000.0

    assert count == 67
    return p
