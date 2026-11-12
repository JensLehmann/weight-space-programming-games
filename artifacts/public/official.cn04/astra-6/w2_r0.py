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

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    opposite = [1, 0, 3, 2]
    neighbours = [(2 + 2 * dy, 2 + 2 * dx)
                  for dy, dx in directions]

    def core(channel, colours, dy=0, dx=0):
        if isinstance(colours, int):
            colours = [colours]
        for colour in colours:
            p["W1"][channel, colour,
                    1 + dy:4 + dy, 1 + dx:4 + dx] = 1
        p["b1"][channel] = -8

    # Centres of canonical 3x3 connector cells.
    core(0, 8)
    core(1, 3)

    for d, (y, x) in enumerate(neighbours):
        p["W1"][2 + d, 0, y, x] = 1

    # Grey endpoints: a grey core with exactly one grey neighbour.
    for d in range(4):
        channel = 6 + d
        p["W1"][channel, 4, 1:4, 1:4] = 1
        for j, (y, x) in enumerate(neighbours):
            p["W1"][channel, 4, y, x] = 1 if j == d else -1
        p["b1"][channel] = -9

    # The first two layers displace query masks by two pixels.
    # A final recurrent readout displacement gives a full grid step.
    for d, (dy, dx) in enumerate(directions):
        core(10 + d, [8, 3], -dy, -dx)

    core(14, 4)
    p["W1"][15, 16, 2, 2] = 1
    p["W1"][16, 17, 2, 2] = 1

    # Black expansion handle attached to a colour-11 wire.
    p["W1"][17, 0, 1:4, 1:4] = 1
    for y, x in neighbours:
        p["W1"][17, 0, y, x] = -1
        p["W1"][17, 11, y, x] = 1
    p["b1"][17] = -9

    # Local directional colour contrasts.
    for d in range(4):
        for side, sign in enumerate((1, -1)):
            channel = 18 + 2 * d + side
            for j, (y, x) in enumerate(neighbours):
                coefficient = sign * (3 if j == d else -1)
                for colour in range(16):
                    p["W1"][channel, colour, y, x] = (
                        coefficient * colour
                    )

    for y, x in neighbours:
        p["W1"][26, 11, y, x] = 1
        p["W1"][27, 11, y, x] = 1
    p["b1"][27] = -1

    # Sparse wire features for fallback clicks.
    for channel, colour in ((28, 14), (29, 15), (31, 12)):
        p["W1"][channel, colour, 1:4, 1:4] = 1
        for y, x in neighbours:
            p["W1"][channel, colour, y, x] = -2
        p["b1"][channel] = -1

    for y, x in neighbours:
        p["W1"][30, 4, y, x] = 1

    def second(out, terms, bias=0):
        for source, weight in terms.items():
            p["W2"][out, source, 1, 1] = weight
        p["b2"][out] = bias

    second(0, {0: 1})
    second(1, {1: 1})

    # Strictly gate black adjacency by connector presence.
    black_neighbours = {2: 1, 3: 1, 4: 1, 5: 1}
    second(2, dict(black_neighbours, **{}) | {0: 4}, -4)
    second(3, dict(black_neighbours, **{}) | {1: 4}, -4)
    second(30, dict(black_neighbours, **{}) | {0: 4}, -5)
    second(31, dict(black_neighbours, **{}) | {1: 4}, -5)

    second(4, {6: 1, 7: 1, 8: 1, 9: 1})
    second(5, {0: 1, 2: -1, 3: -1, 4: -1, 5: -1})
    second(6, {14: 1})
    second(7, {17: 1})
    second(8, {15: 1})
    second(9, {26: 1, 27: -2})
    second(10, {0: 1, 1: 1})

    for d, (dy, dx) in enumerate(directions):
        p["W2"][11 + d, 10 + d, 1 - dy, 1 - dx] = 1

    for d in range(4):
        second(15 + d, {0: 1, 1: 1, 2 + d: 1}, -1)

    # Clipped estimates of nonselected red-terminal body direction.
    for d in range(4):
        terms = {0: 128.0}
        for j in range(4):
            coefficient = (2.0 if j == d else -1.0) / 3.0
            terms[18 + 2 * j] = coefficient
            terms[19 + 2 * j] = coefficient
            terms[2 + j] = -128.0
        second(19 + d, terms, -128)
        second(23 + d, terms, -129)

    second(27, {28: 1, 29: 1, 31: 1})
    second(28, {1: 1, 30: -1})
    second(29, {1: 1, 2: -1, 3: -1, 4: -1, 5: -1})

    for channel in range(19):
        p["W3"][channel, channel] = 1

    p["W3"][2, 30] = -1
    p["W3"][3, 31] = -1

    for d in range(4):
        p["W3"][19 + d, 19 + d] = 1
        p["W3"][19 + d, 23 + d] = -1

    p["W3"][23, 27] = 1
    p["W3"][24, 29] = 1
    p["W3"][25, 2] = 1
    p["W3"][25, 3] = 1
    p["W3"][25, 30] = -1
    p["W3"][25, 31] = -1
    p["W3"][26, 1] = 1
    p["W3"][26, 28] = -1
    p["W3"][27, 28] = 1
    p["W3"][28, 4] = 1
    p["W3"][28, 1] = 1
    p["W3"][29, 5] = 1
    p["W3"][29, 1] = 1

    # Each field uses:
    # reach, reach-excess, arrival-time integral,
    # four translated query readouts, and a current-position readout.
    for base, source in ((0, 28), (8, 29)):
        reach = base
        excess = base + 1
        integral = base + 2

        for out in (reach, excess):
            for dy, dx in [(0, 0)] + directions:
                p["Wh"][out, reach, 1 + dy, 1 + dx] += 1
                p["Wh"][out, excess, 1 + dy, 1 + dx] -= 1
            p["Uh"][out, source] = 1
        p["bh"][excess] = -1

        p["Wh"][integral, integral, 1, 1] = 1
        p["Wh"][integral, reach, 1, 1] = 1
        p["Wh"][integral, excess, 1, 1] = -1

        for d, (dy, dx) in enumerate(directions):
            out = base + 3 + d
            p["Wh"][out, integral, 1 + dy, 1 + dx] = 1
            p["Uh"][out, 11 + d] = 256
            p["bh"][out] = -256

        out = base + 7
        p["Wh"][out, integral, 1, 1] = 1
        p["Uh"][out, 10] = 256
        p["bh"][out] = -256

    def hm(channel):
        return channel

    def ha(channel):
        return 16 + channel

    def fm(channel):
        return 32 + channel

    def fa(channel):
        return 64 + channel

    grey = fm(6)
    gate = 100000.0
    nearest_weight = 0.25

    z = 0
    for d in range(4):
        for base, masked in ((0, True), (8, False)):
            advantage = np.zeros(96, np.float32)
            advantage[ha(base + 3 + d)] = 4096
            advantage[ha(base + 7)] = -4096

            if masked:
                advantage[hm(base + 3 + d)] += nearest_weight
                advantage[hm(base + 7)] -= nearest_weight

                # A translated green query normally incurs three
                # pixels of cost. Refund most, but not all, of it.
                advantage[fa(1)] += 2.4 * 4096
            else:
                # Remove the artificial motion cost of static queries.
                advantage[fa(5)] += 3 * 4096
                advantage[fa(24)] += 3 * 4096
                advantage[fa(3)] += 2.4 * 4096

            for sign in (1.0, -1.0):
                p["Wz"][z] = sign * advantage
                if masked:
                    p["Wz"][z, grey] += gate
                    p["bz"][z] = -gate
                else:
                    p["Wz"][z, grey] -= gate
                p["Wo"][d + 1, z] = sign
                z += 1

    # Current unmodified distance objective for progress detection.
    for out, base, masked in ((16, 0, True), (17, 8, False)):
        p["Wz"][out, fa(10)] = 126 * 4096
        p["Wz"][out, ha(base + 7)] = -4096
        if masked:
            p["Wz"][out, grey] = gate
            p["bz"][out] = -gate
        else:
            p["Wz"][out, grey] = -gate

    p["Wz"][18, fm(7)] = 1

    # Missing selection.
    p["Wz"][19, grey] = 1
    p["Wz"][19, fm(0)] = -1
    p["Wz"][19, fm(1)] = -1

    p["Wz"][20, grey] = -1
    p["Wz"][20, fm(25)] = -1
    p["bz"][20] = 1

    # Switch away from a fully connected selected piece.
    p["Wz"][21, fm(1)] = 1
    p["Wz"][21, fm(0)] = -1
    p["Wz"][21, grey] = 1
    p["bz"][21] = -1

    p["Wz"][22, fm(3)] = 1
    p["Wz"][22, fm(2)] = -1
    p["Wz"][22, fm(5)] = 1
    p["Wz"][22, grey] = -1
    p["bz"][22] = -1

    # Keep expanded branching hubs stationary.
    p["Wz"][23, fa(10)] = 4096
    p["Wz"][23, fm(9)] = 32
    p["Wz"][23, grey] = 32
    p["bz"][23] = -66.5

    # Prefer opposing terminal faces in unmasked scenes.
    for d in range(4):
        out = 24 + d
        p["Wz"][out, fm(15 + d)] = 1
        p["Wz"][out, fm(19 + opposite[d])] = -1
        p["Wz"][out, grey] = -1
        p["Wz"][out, fm(1)] = -1
        p["Wo"][5, out] = 100

    # Unary rotation-sweep memory, reset on selection.
    p["Wm"][0, 0] = 8
    p["Wma"][0, 5] = 16
    p["Wma"][0, 6] = -48
    p["bm"][0] = -4

    for i in range(1, 4):
        p["Wm"][i, i] = 12
        p["Wm"][i, i - 1] = 6
        p["Wma"][i, 5] = 12
        p["Wma"][i, 6] = -48
        p["bm"][i] = -1

    # Expansion latch.
    p["Wm"][4, 4] = 8
    p["Wmz"][4, 18] = 16
    p["Wma"][4, 6] = -48
    p["bm"][4] = -4

    p["Wmz"][5, 16] = 1e-6
    p["Wmz"][5, 17] = 1e-6

    p["bo"][1:5] = -20.15

    for d in range(4):
        action = d + 1
        cell = 6 + d

        # Repeating a non-improving attempted movement is discouraged.
        p["Wm"][cell, 5] = -1e7
        p["Wmz"][cell, 16] = 10
        p["Wmz"][cell, 17] = 10
        p["Wma"][cell, action] = 50000
        p["bm"][cell] = -49997
        p["Wo"][action, 128 + cell] = -10

        # Suppress immediate movement reversals.
        cell = 10 + d
        p["Wma"][cell, action] = 8
        p["bm"][cell] = -4
        p["Wo"][opposite[d] + 1, 128 + cell] = -10

    p["Wo"][5, 18] = 1000

    p["bo"][6] = 4.1
    p["Wo"][6, 128 + 3] = 1
    p["Wo"][6, 128 + 4] = 5
    p["Wo"][6, 19] = 50
    p["Wo"][6, 20] = 50
    p["Wo"][6, 21] = 5
    p["Wo"][6, 22] = 5
    p["Wo"][6, 23] = 100

    p["bo"][0] = -10000
    p["bo"][7] = -10000

    # In grey scenes the second field originates at the selected
    # piece's connectors. Lower integral means greater distance:
    # this breaks repeated selection loops between nearby pieces.
    p["Wc"][0, 4] = 10000
    p["Wc"][0, 5] = 2000
    p["Wc"][0, 24] = 500
    p["Wc"][0, 6] = 100
    p["Wc"][0, 23] = 1
    p["Wc"][0, 8] = 0.01
    p["Wc"][0, 32 + 10] = -0.5

    return p
