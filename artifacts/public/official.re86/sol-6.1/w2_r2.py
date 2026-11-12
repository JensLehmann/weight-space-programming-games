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

    colours = (6, 7, 8, 9, 10, 11, 12, 14)
    cardinal = ((-1, 0), (1, 0), (0, -1), (0, 1))
    diagonal = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    around = cardinal + diagonal
    paint_colours = (8, 9, 11)
    paint_channels = {
        8: (19, 20),
        9: (21, 22),
        11: (23, 28),
    }

    def c1(out, inp, dy, dx, value):
        p["W1"][out, inp, dy + 2, dx + 2] += value

    def c2(out, inp, value, dy=0, dx=0):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    def local(out, inp, value):
        p["W3"][out, inp] += value

    def row(out, inp, value):
        p["W3"][out, 32 + inp] += value

    def rec(out, inp, value, dy=0, dx=0):
        p["Wh"][out, inp, dy + 1, dx + 1] += value

    # Target centres: require all four cardinal or diagonal
    # border pixels. An uncovered target contributes two;
    # a target crossed by a stroke generally contributes one.
    for j, colour in enumerate(colours):
        for out, offsets in (
            (2 * j, diagonal),
            (2 * j + 1, cardinal),
        ):
            c1(out, colour, 0, 0, 8.0)
            for dy, dx in offsets:
                c1(out, 4, dy, dx, 1.0)
            p["b1"][out] = -11.0

    # Marker-local weighted colour and neighbour counts.
    c1(16, 0, 0, 0, 2.0)
    p["b1"][16] = -2.0
    c1(17, 0, 0, 0, 9.0)
    p["b1"][17] = -9.0
    c1(18, 0, 0, 0, 9.0)
    p["b1"][18] = -9.0

    for colour in colours:
        for dy, dx in around:
            c1(16, colour, dy, dx, colour / 64.0)
            c1(17, colour, dy, dx, 1.0)
        for dy, dx in cardinal:
            c1(18, colour, dy, dx, 1.0)

    c1(24, 0, 0, 0, 1.0)
    c1(26, 16, 0, 0, 1.0)
    c1(27, 17, 0, 0, 1.0)

    for colour in colours:
        c1(25, colour, 0, 0, colour / 16.0)
        c1(30, colour, 0, 0, colour / 16.0)

    # Hollow-centre fallback excludes targets and paint stations.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            c1(30, 2, dy, dx, -16.0)
            c1(30, 4, dy, dx, -16.0)

    # Paint coordinates are encoded as 1 + coordinate/128,
    # at coloured interior pixels immediately below a border.
    for colour in paint_colours:
        rch, cch = paint_channels[colour]
        for out, coordinate_input in ((rch, 16), (cch, 17)):
            c1(out, coordinate_input, 0, 0, 1.0 / 128.0)
            c1(out, colour, 0, 0, 10.0)
            c1(out, 2, -1, 0, 10.0)
            p["b1"][out] = -19.0

    # Second convolution.
    for j in range(8):
        c2(j, 2 * j, 1.0)
        c2(j, 2 * j + 1, 1.0)

    c2(8, 16, 1.0)
    c2(9, 17, 1.0)
    c2(10, 18, 1.0)

    # Correct the colour average for a two-neighbour line.
    # Difference of these units equals the weighted key only
    # when the neighbour count is two.
    for out in (11, 12):
        c2(out, 17, -1.0)
        p["b2"][out] = 2.0
    c2(11, 16, 1.0)

    # Binary neighbour-presence and cardinal-presence pairs.
    for positive, negative, source in (
        (13, 14, 17),
        (15, 16, 18),
    ):
        c2(positive, source, 2.0)
        c2(negative, source, 2.0)
        p["b2"][negative] = -1.0

    for out, source in (
        (17, 24), (18, 25), (19, 26),
        (20, 27), (21, 30),
    ):
        c2(out, source, 1.0)

    paint_f2 = {}
    for j, colour in enumerate(paint_colours):
        rch, cch = paint_channels[colour]
        rout, cout = 22 + 2 * j, 23 + 2 * j
        paint_f2[colour] = (rout, cout)
        c2(rout, rch, 1.0)
        c2(cout, cch, 1.0)

    # Axial feature layout:
    # 0,1 selected colour and hollow-centre fallback;
    # 2 local colour; 3 target strength; 4 geometric mode;
    # 5,6 marker coordinates; 7 paint-8 presence;
    # 8..15 individual target strengths;
    # 16..21 paint coordinates;
    # 22,23 paint-9 and paint-11 presence;
    # 24,25 spatial coordinates;
    # 26..28 uncovered targets of colours 8,9,11;
    # 29 visible marker.
    row(0, 8, 1.0)
    row(0, 11, 1.0)
    row(0, 12, -1.0)

    row(1, 21, 1.0)
    row(1, 13, -2.0)
    row(1, 14, 2.0)
    row(1, 17, 2.0)
    p["b3"][1] = -2.0

    local(2, 18, 1.0)

    for j in range(8):
        local(3, j, 1.0)
        local(8 + j, j, 1.0)

    row(4, 17, 1.0)
    row(4, 15, -1.0)
    row(4, 16, 1.0)

    local(5, 17, 1.0)
    local(5, 19, 1.0)
    p["b3"][5] = -1.0

    local(6, 17, 1.0)
    local(6, 20, 1.0)
    p["b3"][6] = -1.0

    local(24, 19, 1.0)
    local(25, 20, 1.0)
    local(29, 17, 1.0)

    paint_f3 = {}
    presence_f3 = {8: 7, 9: 22, 11: 23}
    uncovered_f3 = {8: 26, 9: 27, 11: 28}

    for j, colour in enumerate(paint_colours):
        rin, cin = paint_f2[colour]
        rout, cout = 16 + 2 * j, 17 + 2 * j
        paint_f3[colour] = (rout, cout)

        local(rout, rin, 1.0)
        local(cout, cin, 1.0)
        p["b3"][rout] = -1.0
        p["b3"][cout] = -1.0

        presence = presence_f3[colour]
        local(presence, rin, 1.0)
        local(presence, cin, 1.0)
        local(presence, 19, -1.0 / 128.0)
        local(presence, 20, -1.0 / 128.0)

        uncovered = uncovered_f3[colour]
        local(uncovered, colours.index(colour), 1.0)
        p["b3"][uncovered] = -1.0

    # Broadcast selected colour vertically.
    for out, dy in ((0, -1), (1, 1)):
        p["Uh"][out, 0] = 1.0
        p["Uh"][out, 1] = 1.0
        rec(out, out, 1.0, dy, 0)

    # Positive colour mismatch.
    rec(2, 0, 1.0)
    rec(2, 1, 1.0)
    p["Uh"][2, 0] = -1.0
    p["Uh"][2, 1] = -1.0
    p["Uh"][2, 2] = -1.0

    # Matching-colour target strength.
    p["Uh"][3, 3] = 1.0
    rec(3, 0, 64.0)
    rec(3, 1, 64.0)
    rec(3, 2, -128.0)
    p["Uh"][3, 0] = -64.0
    p["Uh"][3, 1] = -64.0
    p["Uh"][3, 2] = -64.0

    # Broadcast cardinal/diagonal mode.
    for out, dy in ((4, -1), (5, 1)):
        p["Uh"][out, 4] = 1.0
        rec(out, out, 1.0, dy, 0)

    # Directional target-voting fields.
    for out, (dy, dx) in enumerate(around, start=6):
        rec(out, out, 1.0, dy, dx)
        rec(out, 3, 1.0)

        if out < 10:
            rec(out, 4, -100.0)
            rec(out, 5, -100.0)
            p["Uh"][out, 4] = 100.0
        else:
            rec(out, 4, 100.0)
            rec(out, 5, 100.0)
            p["Uh"][out, 4] = -100.0
            p["bh"][out] = -100.0

    # Integer vote plus fractional coordinate.
    for out, feature in ((14, 24), (15, 25)):
        for ray in range(6, 14):
            rec(out, ray, 1.0)
        rec(out, 3, -3.0)
        p["Uh"][out, feature] = 63.0 / 128.0

    # Pooled vector:
    # [h.max, h.mean, f3.max, f3.mean].
    p["Wz"][0, 14] = 1.0
    p["Wz"][1, 15] = 1.0
    p["Wz"][2, 32 + 5] = 1.0
    p["Wz"][3, 32 + 6] = 1.0

    # Hinge decoding of the integer part of the vote.
    unit = 4
    floor_units = []
    for maximum_channel in (14, 15):
        terms = []
        for k in range(1, 25):
            p["Wz"][unit, maximum_channel] = 2.0
            p["bz"][unit] = 1.0 - 2.0 * k
            terms.append((unit, 1.0))
            unit += 1

            p["Wz"][unit, maximum_channel] = 2.0
            p["bz"][unit] = -2.0 * k
            terms.append((unit, -1.0))
            unit += 1
        floor_units.append(terms)

    scale = 128.0 / 3.0
    for action, axis, sign in (
        (1, 0, 1.0), (2, 0, -1.0),
        (3, 1, 1.0), (4, 1, -1.0),
    ):
        p["Wo"][action, axis] = -sign * scale
        p["Wo"][action, 2 + axis] = sign * 21.0
        for hinge, coefficient in floor_units[axis]:
            p["Wo"][action, hinge] = sign * scale * coefficient

    # Paint navigation is enabled only when the selected colour
    # has no matching targets. Cardinal objects favour paint 8,
    # diagonal/hollow objects favour paint 9.
    for colour in paint_colours:
        rfeature, cfeature = paint_f3[colour]
        for action, axis, sign in (
            (1, 0, 1.0), (2, 0, -1.0),
            (3, 1, 1.0), (4, 1, -1.0),
        ):
            u = unit
            unit += 1

            target_feature = rfeature if axis == 0 else cfeature
            p["Wz"][u, 32 + 5 + axis] = sign * 21.0
            p["Wz"][u, 32 + target_feature] = -sign * 2688.0

            p["Wz"][u, 32 + uncovered_f3[colour]] = 100.0
            p["Wz"][u, 32 + presence_f3[colour]] = 50.0
            p["Wz"][u, 3] = -200.0
            p["bz"][u] = -200.5

            if colour == 9:
                p["Wz"][u, 32 + 4] = 100.0
                p["bz"][u] -= 100.0
            elif colour == 8:
                p["Wz"][u, 32 + 4] = -100.0
            else:
                p["Wz"][u, 32 + uncovered_f3[8]] = -200.0
                p["Wz"][u, 32 + uncovered_f3[9]] = -200.0

            p["Wo"][action, u] = 100.0

    # If all matching targets are covered, preserve the object
    # and select the next one. The interval excludes the
    # no-matching-target case when paint stations are present.
    u = unit
    unit += 1
    p["Wz"][u, 3] = -1.0
    p["bz"][u] = 1.5
    p["Wo"][5, u] = 1000.0

    # Cancel that switch preference for a colour with no targets
    # when a relevant paint station is visible.
    u = unit
    unit += 1
    p["Wz"][u, 3] = -4.0
    p["bz"][u] = 1.0
    p["Wo"][5, u] = -1500.0

    # A marker can be hidden under the bottom UI row.
    u = unit
    unit += 1
    p["Wz"][u, 32 + 29] = -1.0
    p["bz"][u] = 1.0
    p["Wo"][5, u] = 100000.0

    # Small reversal penalty.
    opposite = {1: 2, 2: 1, 3: 4, 4: 3}
    for action in range(1, 5):
        p["Wma"][action, action] = 1.0
        p["Wo"][action, 128 + opposite[action]] = -0.04

    p["bo"][:] = -100000.0
    p["bo"][1:5] = -0.5
    p["bo"][5] = 0.0

    return p
