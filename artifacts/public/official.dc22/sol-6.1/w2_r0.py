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
    p = {k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()}

    def px(unit, colour, dy=0, dx=0, weight=1.0):
        p["W1"][unit, colour, dy + 2, dx + 2] += weight

    def pattern(unit, colour, locations, weight=1.0):
        for dy, dx in locations:
            px(unit, colour, dy, dx, weight)

    square = [(0, 0), (0, 1), (1, 0), (1, 1)]
    terrain = [1, 2, 6, 7, 8, 9, 11, 12, 13, 14, 15]

    # Full-footprint support for an actual two-pixel move.
    for colour in terrain:
        pattern(0, colour, square)
    p["b1"][0] = -3.0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            px(0, 0, dy, dx, -4.0)

    # Isolated goal.
    pattern(1, 11, square)
    pattern(1, 11, [(-1, 0), (0, -1), (2, 0), (0, 2)], -4.0)
    p["b1"][1] = -3.0

    # Portal footprint.
    for colour in (6, 7):
        pattern(2, colour, square)
    p["b1"][2] = -3.0

    # Destination indicators for up, down, left, right.
    markers = [
        ((2, 0), [(1, 0), (2, -1)]),
        ((-1, 0), [(0, 0), (-1, -1)]),
        ((0, 2), [(0, 1), (-1, 2)]),
        ((0, -1), [(0, 0), (-1, -1)]),
    ]
    for unit, (positive, negative) in enumerate(markers, 3):
        px(unit, 14, *positive)
        pattern(unit, 14, negative, -1.0)

    # Orange gate phases.
    pattern(7, 9, square)
    p["b1"][7] = -3.0
    pattern(8, 9, [(0, 0), (1, 1)])
    pattern(8, 4, [(0, 1), (1, 0)])
    p["b1"][8] = -3.0
    pattern(9, 9, [(0, 1), (1, 0)])
    pattern(9, 4, [(0, 0), (1, 1)])
    p["b1"][9] = -3.0

    # Paired ramps for any pink pixel in the footprint.
    pattern(10, 15, square)
    pattern(11, 15, square)
    p["b1"][11] = -1.0

    # Pivot footprint and coordinate-valued pivot features.
    pattern(13, 13, square)
    p["b1"][13] = -3.0
    for unit, coordinate, inverse in (
        (14, 16, False), (15, 16, True),
        (16, 17, False), (17, 17, True),
    ):
        pattern(unit, 13, square, 2.0)
        px(unit, coordinate, weight=-1.0 if inverse else 1.0)
        if inverse:
            px(unit, 18)
        p["b1"][unit] = -8.0

    # UI column marker.
    pattern(18, 5, [(-1, 0), (0, 0), (1, 0)])
    p["b1"][18] = -2.0

    # Switch interiors, excluding small control squares.
    interior = [(dy, dx)
                for dy in (-1, 0, 1)
                for dx in (-2, -1, 0, 1, 2)]
    interior += [(-2, 0), (2, 0)]
    for unit, colour in zip(range(19, 25), (8, 9, 15, 6, 2, 11)):
        pattern(unit, colour, interior)
        p["b1"][unit] = -16.0

    px(25, 18)
    px(26, 16)
    px(27, 17)

    # Player's current top-left cell.
    px(28, 14)
    pattern(28, 14, [(-1, 0), (0, -1)], -1.0)

    # Switch-unlocking pickup.
    pattern(29, 15, [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])
    pattern(29, 2, [(0, 0), (-1, 0)])
    p["b1"][29] = -6.0

    # Prospective support: clip the presence of any terrain pixel.
    for unit, bias in ((30, 0.0), (31, -1.0)):
        for colour in terrain:
            pattern(unit, colour, square)
        p["b1"][unit] = bias
        for dy, dx in square:
            px(unit, 0, dy, dx, -4.0)

    def cp2(out, inp, weight=1.0):
        p["W2"][out, inp, 1, 1] += weight

    for unit in range(7):
        cp2(unit, unit)
    for inp in (7, 8, 9):
        cp2(7, inp)

    cp2(8, 10)
    cp2(8, 11, -1.0)
    # The pickup is not a locked pink bridge.
    p["W2"][8, 29, :, :] = -16.0

    for out, inp in zip(range(9, 14), range(13, 18)):
        cp2(out, inp)
    cp2(14, 18)
    for out, inp in zip(range(15, 21), range(19, 25)):
        cp2(out, inp)
    for inp in range(3, 7):
        cp2(21, inp)
    for out, inp in (
        (22, 25), (23, 26), (24, 27),
        (25, 28), (26, 29), (27, 30),
    ):
        cp2(out, inp)
    cp2(27, 31, -1.0)

    # Orange and pink neighbourhoods for boundary switch requests.
    for inp in (8, 9):
        p["W2"][30, inp, :, :] = 1.0
    p["W2"][31, 10, :, :] = 1.0
    p["W2"][31, 11, :, :] = -1.0
    p["W2"][31, 29, :, :] = -16.0

    def lin3(out, terms, bias=0.0):
        for inp, weight in terms:
            p["W3"][out, inp] += weight
        p["b3"][out] = bias

    lin3(0, [(0, 1.0)])
    lin3(1, [(7, 1.0)])
    lin3(2, [(8, 1.0)])

    # Hypothetical pivot axes, only where real terrain is absent.
    lin3(3, [(64 + 9, 1.0), (27, -1.0)])
    lin3(4, [(32 + 9, 1.0), (27, -1.0)])

    # Paired ramps yield saturated prospective bridge-end tests.
    radius = 10.0
    for out, offset in ((5, .1), (6, .9)):
        lin3(out, [(23, 63.0), (64 + 10, -63.0),
                   (64 + 9, 128.0)],
             -radius - 128.0 - offset)
    for out, offset in ((7, .1), (8, .9)):
        lin3(out, [(23, -63.0), (64 + 11, -63.0),
                   (64 + 9, 128.0)],
             63.0 - radius - 128.0 - offset)
    for out, offset in ((9, .1), (10, .9)):
        lin3(out, [(24, 63.0), (32 + 12, -63.0),
                   (32 + 9, 128.0)],
             -radius - 128.0 - offset)
    for out, offset in ((11, .1), (12, .9)):
        lin3(out, [(24, -63.0), (32 + 13, -63.0),
                   (32 + 9, 128.0)],
             63.0 - radius - 128.0 - offset)

    lin3(13, [(64 + 14, 1.0)])
    lin3(14, [(1, 1.0)])
    lin3(15, [(26, 1.0)])
    for i in range(4):
        lin3(16 + i, [(3 + i, 1.0)])

    for out, inp in ((20, 15), (21, 16), (22, 17), (23, 18)):
        lin3(out, [(inp, 1.0), (64 + 14, 1.0)], -1.0)

    # Blue requests at hypothetical-axis destinations.
    lin3(24, [(21, 1.0), (64 + 9, 1.0),
              (0, -1.0), (27, -1.0)], -1.0)
    lin3(25, [(21, 1.0), (32 + 9, 1.0),
              (0, -1.0), (27, -1.0)], -1.0)

    request_gate = 256.0
    lin3(26, [(21, request_gate), (30, 1.0),
              (0, -request_gate)], -request_gate)
    lin3(27, [(21, request_gate), (31, 1.0),
              (0, -request_gate)], -request_gate)

    lin3(28, [(2, 1.0), (64 + 14, -16.0)])
    lin3(29, [(27, 1.0)])
    lin3(30, [(25, 1.0)])
    lin3(31, [(15, 1.0), (16, 1.0), (64 + 18, -1.0)])

    # h0: obstacle map for prospective routes.
    p["bh"][0] = 2.0
    p["Uh"][0, 29] = -4.0
    p["Uh"][0, 3] = -2.0
    p["Uh"][0, 4] = -2.0
    p["Uh"][0, 2] = 3.0
    for first, second in ((5, 6), (7, 8), (9, 10), (11, 12)):
        p["Uh"][0, first] = 2.5
        p["Uh"][0, second] = -2.5
    p["Uh"][0, 13] = 16.0
    p["Wh"][0, 14, 1, 1] = -1.0
    p["Wh"][0, 15, 1, 1] = 1.0

    cardinal = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    surrounding = [(dy, dx) for dy in (-1, 0, 1)
                   for dx in (-1, 0, 1)]

    def wave(first, second, neighbourhood):
        for out in (first, second):
            for dy, dx in neighbourhood:
                p["Wh"][out, first, dy + 1, dx + 1] += 1.0
                p["Wh"][out, second, dy + 1, dx + 1] -= 1.0
        p["bh"][second] -= 1.0

    # Differences of paired channels are clipped wavefronts.
    wave(1, 2, cardinal)       # goal or pickup
    wave(3, 4, cardinal)       # portals
    wave(10, 11, surrounding) # blue-switch request
    wave(12, 13, surrounding) # orange-switch request
    wave(14, 15, surrounding) # pink-switch availability

    for out in (1, 2):
        p["Uh"][out, 14] = 1.0
        p["Uh"][out, 15] = 1.0
        p["Wh"][out, 0, 1, 1] = -32.0
    for out in (3, 4):
        p["Uh"][out, 28] = 1.0
        p["Wh"][out, 0, 1, 1] = -32.0

    for out in (10, 11):
        p["Uh"][out, 24] = 1.0
        p["Uh"][out, 25] = 1.0
        for feature in (21, 22, 23):
            p["Uh"][out, feature] = -32.0
    for out in (12, 13):
        p["Uh"][out, 26] = 1.0
        for feature in (20, 22, 23):
            p["Uh"][out, feature] = -32.0
    for out in (14, 15):
        p["Uh"][out, 22] = 1.0
        p["Uh"][out, 31] = 1.0

    # Integrate arrivals at candidate moves and click-intent cells.
    priority = 256.0
    inhibit = 65536.0
    for out in range(5, 10):
        p["Wh"][out, out, 1, 1] = 1.0
        p["Wh"][out, 1, 1, 1] = priority
        p["Wh"][out, 2, 1, 1] = -priority
        p["Wh"][out, 3, 1, 1] = 1.0
        p["Wh"][out, 4, 1, 1] = -1.0

    for i in range(4):
        out = 5 + i
        p["Uh"][out, 16 + i] = inhibit
        p["Uh"][out, 0] = inhibit
        p["bh"][out] = -2.0 * inhibit

    for feature in range(16, 20):
        p["Uh"][9, feature] = inhibit
    p["Uh"][9, 0] = -inhibit
    p["Uh"][9, 30] = 2.0 * inhibit
    p["bh"][9] = -inhibit

    for i in range(5):
        p["Wz"][i, 5 + i] = 1.0
    p["Wz"][5, 9] = 1.0
    p["Wz"][6, 9] = 1.0
    p["bz"][6] = -1.0

    p["bo"][:] = -1000000.0
    for action in range(1, 5):
        p["Wo"][action, action - 1] = 1.0
        p["bo"][action] = -.05
    p["Wo"][6, 4] = 1.0
    p["Wo"][6, 5] = -.15
    p["Wo"][6, 6] = .15
    p["bo"][6] = 0.0

    # Only switch interiors receive a large click score.
    for feature in (20, 21, 22, 23):
        p["Wc"][0, feature] = 20.0
    p["Wc"][0, 22] += .2
    p["Wc"][0, 23] += .1
    p["Wc"][0, 42] = 1.0
    p["Wc"][0, 43] = -1.0
    p["Wc"][0, 44] = 1.0
    p["Wc"][0, 45] = -1.0

    return p
