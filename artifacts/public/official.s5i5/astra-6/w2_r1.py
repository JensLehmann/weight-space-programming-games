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

    colors = (7, 8, 9, 10, 11, 12, 14)
    cardinal = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def w1(out, inp, dy=0, dx=0, value=1.0):
        p["W1"][out, inp, dy + 2, dx + 2] += value

    def w2(out, inp, dy=0, dx=0, value=1.0):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    def local(out, inp, value=1.0):
        p["W3"][out, inp] += value

    def row(out, inp, value=1.0):
        p["W3"][out, 32 + inp] += value

    def col(out, inp, value=1.0):
        p["W3"][out, 64 + inp] += value

    # Layer 1:
    # 0..6: actuator colour.
    # 7..13: direct isolated-dot contact.
    # 14..20: material belonging to a rod longer than one cell.
    for k, color in enumerate(colors):
        w1(k, color)

        out = 7 + k
        w1(out, color, value=4.0)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                w1(out, 13, dy, dx, -1.0)
        for dy, dx in cardinal:
            w1(out, 13, dy, dx, 2.0)
        p["b1"][out] = -4.0

        out = 14 + k
        w1(out, color, value=4.0 / 3.0)
        for dy, dx in cardinal:
            w1(out, color, dy, dx, 1.0 / 3.0)
            w1(out, color, 2 * dy, 2 * dx, 1.0 / 3.0)
        p["b1"][out] = -7.0 / 3.0

        # Exclude the control panel from rod detection.
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                w1(out, 2, dy, dx, -8.0)

    # Four shifted dot detectors. An isolated dot gives 1; the centre
    # of a filled target gives 0.4. Ring arms are rejected.
    shifts = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for j, (sy, sx) in enumerate(shifts):
        out = 21 + j
        w1(out, 13, sy, sx)
        for dy, dx in cardinal:
            w1(out, 13, sy + dy, sx + dx, -0.15)
        for dy in (-1, 1):
            for dx in (-1, 1):
                w1(out, 13, sy + dy, sx + dx, -1.0)

    # Left/right/up/down resize-control glyphs.
    for j, (oy, ox) in enumerate(
        ((0, -1), (0, 1), (-1, 0), (1, 0))
    ):
        out = 25 + j
        py, px = -ox, oy
        for color in colors:
            w1(out, color)
            w1(out, color, py, px)
            w1(out, color, -py, -px)
        w1(out, 2, oy, ox)
        w1(out, 2, 2 * oy, 2 * ox)
        w1(out, 4, -oy, -ox)
        w1(out, 4, -2 * oy, -2 * ox)
        p["b1"][out] = -5.0

    # Unfilled target centre.
    for dy, dx in cardinal:
        w1(29, 13, dy, dx)
    w1(29, 13, value=-1.0)
    p["b1"][29] = -3.0

    # Grey-bar / wall proximity. Requiring a full six-neighbour
    # material cross largely rejects contact at a bar's longitudinal
    # end while detecting walls normal to the bar. Magenta is allowed
    # in the cross because target rings can overlay grey barriers.
    w1(30, 1, value=100.0)
    for dy, dx in cardinal:
        for distance in (1, 2):
            w1(30, 1, distance * dy, distance * dx)
            w1(30, 13, distance * dy, distance * dx)
        w1(30, 15, 2 * dy, 2 * dx, 16.0)
    p["b1"][30] = -121.5

    # Rotation-control centre.
    for color in colors:
        w1(31, color)
        for dy, dx in cardinal:
            w1(31, color, dy, dx)
    for dy in (-1, 1):
        for dx in (-1, 1):
            w1(31, 4, dy, dx)
    for dy, dx in cardinal:
        w1(31, 2, 2 * dy, 2 * dx)
    p["b1"][31] = -12.0

    # Choose one representation of each accessible dot offset.
    # The union is a 5x5 neighbourhood without its four outer corners.
    dot_taps = {}
    for j, (sy, sx) in enumerate(shifts):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                offset = (sy + dy, sx + dx)
                if offset not in dot_taps:
                    dot_taps[offset] = (21 + j, dy, dx)

    # Layer 2:
    # 0..6: dot-associated colour.
    # 7..13: grey joints, with extra strength for long attached rods.
    # 14..20: contacts between different actuator colours.
    # 21..27: colour-labelled controls.
    #
    # A raw-grey plane is reconstructed from the input indirectly:
    # the constant-minus-colour/background construction below uses
    # the fact that all non-grey visible palette categories can be
    # detected by a separate first-layer bookkeeping plane.
    #
    # Use channel 30 of layer 1 for collision detection as above.
    # Grey selection for joints instead uses the absence of actuator
    # colour at the centre together with nearby long-rod material.
    # To make this exact, reserve a raw-grey component in channel 31
    # using a disjoint activation amplitude. Rotation glyphs remain
    # at amplitude 1 and grey pixels at amplitude 32.
    #
    # Both predicates cannot be OR-ed in one affine/ReLU unit, so use
    # channel 31 exclusively for raw grey. Rotation glyphs are omitted
    # from this revision's click candidates.
    p["W1"][31].fill(0.0)
    p["b1"][31] = 0.0
    w1(31, 1)

    for k in range(7):
        out = k
        w2(out, k, value=8.0)
        for inp, dy, dx in dot_taps.values():
            w2(out, inp, dy, dx)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                w2(out, 7 + k, dy, dx, -0.04)
        p["b2"][out] = -8.0

        # Grey-contact values are <=3 for a one-cell actuator.
        # Long-rod evidence lifts them above the retraction threshold.
        out = 7 + k
        w2(out, 31, value=128.0)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                w2(out, k, dy, dx)
                w2(out, 14 + k, dy, dx, 16.0)
        p["b2"][out] = -128.0

        out = 14 + k
        w2(out, k, value=32.0)
        for other in range(7):
            if other != k:
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy or dx:
                            w2(out, other, dy, dx)
        p["b2"][out] = -32.0

        out = 21 + k
        w2(out, k)
        for inp in (25, 26, 27, 28):
            w2(out, inp)
        p["b2"][out] = -1.0

    w2(28, 29)
    for dy, dx in cardinal:
        w2(28, 29, dy, dx)

    w2(29, 25)
    w2(29, 27)
    w2(30, 26)
    w2(30, 28)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            w2(31, 30, dy, dx)

    # Layer 3:
    # 0..6: dot/target movement sources.
    # 7..13: low-priority parent-extension sources.
    # 14..20: barrier-retraction sources.
    # 21..27: labelled controls; 28/29: control halves.
    for k in range(7):
        out = k
        local(out, k)
        row(out, 28, 0.45)
        col(out, 28, 0.45)
        p["b3"][out] = -(1.0 - 2.0e-5)

        # An actuator touching another colour is a possible parent.
        # A grey joint on the same rod axis identifies the barrier's
        # own actuator instead and suppresses this fallback.
        out = 7 + k
        local(out, 14 + k)
        row(out, 7 + k, -1024.0)
        col(out, 7 + k, -1024.0)
        row(out, k, -16.0)
        col(out, k, -16.0)

        out = 14 + k
        local(out, 7 + k)
        row(out, 31, -1024.0)
        col(out, 31, -1024.0)
        row(out, k, -1024.0)
        col(out, k, -1024.0)
        p["b3"][out] = -3.1

        local(21 + k, 21 + k)

    local(28, 29)
    local(29, 30)

    p["b3"][30] = 1.0
    for k in range(7):
        local(30, 21 + k, -1.0)

    # Seven extension and seven retraction diffusion fields.
    gate = float(2 ** 28)
    for k in range(7):
        grow = k
        shrink = 7 + k

        p["Wh"][grow, grow, :, :] = 1.0 / 9.0
        p["Wh"][shrink, shrink, :, :] = 1.0 / 9.0

        p["Uh"][grow, k] = 1.0
        p["Uh"][grow, 7 + k] = 1.0e-12
        p["Uh"][shrink, 14 + k] = 4096.0

        for other in range(7):
            if other != k:
                p["Uh"][grow, 21 + other] = -gate
                p["Uh"][shrink, 21 + other] = -gate

        p["Uh"][grow, 28] = -gate
        p["Uh"][shrink, 29] = -gate

    # Penalize non-controls instead of adding a large control bonus,
    # preserving the small diffusion scores of distant candidates.
    p["Wc"][0, 30] = -float(2 ** 40)
    p["Wc"][0, 29] = 1.0e-30
    p["Wc"][0, 32:46] = 1.0

    p["bo"][:] = -10.0
    p["bo"][6] = 10.0

    return p
