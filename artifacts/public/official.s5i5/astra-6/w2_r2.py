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

    colors = (7, 8, 9, 10, 11, 12, 14)
    cardinal = ((-1, 0), (1, 0), (0, -1), (0, 1))
    diagonal = ((-1, -1), (-1, 1), (1, -1), (1, 1))

    def w1(out, inp, dy=0, dx=0, value=1.0):
        p["W1"][out, inp, dy + 2, dx + 2] += value

    def w2(out, inp, dy=0, dx=0, value=1.0):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    def w3(out, inp, value=1.0, pool=0):
        p["W3"][out, inp + 32 * pool] += value

    def project(out, inp, value):
        # Row/column projections, subtracting their local intersection.
        w3(out, inp, value, 1)
        w3(out, inp, value, 2)
        w3(out, inp, -value, 0)

    # ---------------------------------------------------------------
    # First convolution
    #
    #  0:7   unit dot-colour indicators
    #  7:14  another-colour contact at a dot
    # 14:21  positive control halves
    # 21:28  negative control halves
    # 28     normalized diagonal coordinate
    # 29     unoccupied goal centre
    # 30     goal coordinate plus an offset
    # 31     negative-coordinate direction cue
    # ---------------------------------------------------------------
    for i, color in enumerate(colors):
        w1(i, 13, value=4.0)
        for dy, dx in cardinal:
            w1(i, color, dy, dx)
            w1(i, 3, dy, dx)
        for dy, dx in diagonal:
            w1(i, 13, dy, dx, -4.0)
        p["b1"][i] = -7.0

        contact = 7 + i
        w1(contact, 13, value=16.0)
        for dy, dx in cardinal:
            w1(contact, color, 2 * dy, 2 * dx, 4.0)
            w1(contact, color, dy, dx, -4.0)
            w1(contact, 13, dy, dx, -8.0)
        for dy, dx in diagonal:
            w1(contact, 13, dy, dx, -16.0)
        p["b1"][contact] = -19.0

        for out, sign in ((14 + i, 1.0), (21 + i, -1.0)):
            for dy, dx in cardinal:
                w1(out, color, dy, dx, 6.0)
                w1(out, 2, 2 * dy, 2 * dx, 0.5)

            w1(out, 4, value=6.0)
            # Exclude the solid-colour centres of rotation controls.
            w1(out, color, value=-4.0)

            w1(out, color, 0, 1, sign)
            w1(out, color, 1, 0, sign)
            w1(out, color, 0, -1, -sign)
            w1(out, color, -1, 0, -sign)
            p["b1"][out] = -24.0

    w1(28, 16, value=0.5)
    w1(28, 17, value=0.5)

    for dy, dx in cardinal:
        w1(29, 13, dy, dx)
        w1(30, 13, dy, dx, 8.0)
    w1(29, 13, value=-2.0)
    p["b1"][29] = -3.0
    w1(30, 13, value=-16.0)
    w1(30, 16, value=0.5)
    w1(30, 17, value=0.5)
    p["b1"][30] = -28.0

    w1(31, 13, value=4.0)
    for color in colors:
        w1(31, color, 0, 2)
        w1(31, color, 2, 0)
    w1(31, 3, 0, 1)
    w1(31, 3, 1, 0)
    for dy, dx in diagonal:
        w1(31, 13, dy, dx, -4.0)
    p["b1"][31] = -4.0

    # Cross-correlation transport: downward drift, horizontal spread,
    # and total mass 31/32.
    transport = np.zeros((3, 3), np.float32)
    transport[0, 0] = 15.0 / 64.0
    transport[0, 2] = 15.0 / 64.0
    transport[1, 0] = 1.0 / 4.0
    transport[1, 2] = 1.0 / 4.0

    offset = 32.0

    # ---------------------------------------------------------------
    # Second convolution
    #
    #  0:7   positive-direction sources
    #  7:14  negative-direction sources
    # 14:28  positive/negative decomposition of (I-A)d
    # 28     goal indicator
    # 29     goal coordinate
    # 30     dot coordinate
    # ---------------------------------------------------------------
    for i in range(7):
        w2(i, i)
        w2(i, 31, value=-1.0)

        w2(7 + i, i)
        w2(7 + i, 31)
        p["b2"][7 + i] = -1.0

        # A contact suppresses the original dot-colour request.
        # Its direction cue selects the replacement request.
        for j in range(7):
            w2(i, 7 + j, value=-2.0)
            w2(7 + i, 7 + j, value=-2.0)
        w2(i, 7 + i, value=3.0)
        w2(7 + i, 7 + i, value=3.0)

        cp, cn = 14 + i, 21 + i
        positive_button, negative_button = 14 + i, 21 + i

        w2(cp, negative_button, value=offset)
        w2(cn, positive_button, value=offset)

        for yy in range(3):
            for xx in range(3):
                a = float(transport[yy, xx])
                if a:
                    w2(cp, positive_button, yy - 1, xx - 1,
                       offset * a)
                    w2(cn, negative_button, yy - 1, xx - 1,
                       offset * a)

    w2(28, 29)
    w2(29, 30)
    w2(29, 29, value=-4.0)

    w2(30, 28)
    p["b2"][30] = -2.0
    for i in range(7):
        w2(30, i, value=2.0)

    # ---------------------------------------------------------------
    # Axial layer
    #
    # Each colour has:
    #   ReLU(G_positive + E), ReLU(G_positive),
    #   ReLU(G_negative + E), ReLU(G_negative).
    #
    # Their differences eliminate gate offsets exactly. A common lift
    # at correction pixels permits injection of the signed control
    # offset without changing the colour-dependent geometric signal.
    # ---------------------------------------------------------------
    color_gate = 12.0
    target_gate = 4.0
    gate_bias = -13.0
    lift = 32.0

    for i in range(7):
        for branch in range(4):
            out = 4 * i + branch
            source_channel = i if branch < 2 else 7 + i
            has_error = branch in (0, 2)
            correction_sign = (1.0, -1.0, -1.0, 1.0)[branch]

            project(out, source_channel, color_gate)
            project(out, 28, target_gate)
            p["b3"][out] = gate_bias

            if has_error:
                project(out, 29, 1.0)
                project(out, 30, -1.0)

            w3(out, 14 + i, lift + 0.25 * correction_sign)
            w3(out, 21 + i, lift - 0.25 * correction_sign)

        hp, hn = 2 * i, 2 * i + 1
        for branch, sign in enumerate((1.0, -1.0, -1.0, 1.0)):
            p["Uh"][hp, 4 * i + branch] = sign
            p["Uh"][hn, 4 * i + branch] = -sign

        p["Wh"][hp, hp] = transport
        p["Wh"][hp, hn] = -transport
        p["Wh"][hn, hp] = -transport
        p["Wh"][hn, hn] = transport

        p["Wc"][0, 32 + hp] = -1.0
        p["Wc"][0, 32 + hn] = -1.0

    # Correction for the common, unoriented displacement field.
    # At control-correction pixels both orientation branches are
    # lifted, contributing two copies of E. These units remove them.
    for out in (28, 29):
        for i in range(7):
            w3(out, 14 + i, lift)
            w3(out, 21 + i, lift)
        project(out, 28, target_gate)
        p["b3"][out] = gate_bias

    project(28, 29, 1.0)
    project(28, 30, -1.0)

    for i in range(7):
        for branch, sign in enumerate((1.0, -1.0, 1.0, -1.0)):
            p["Uh"][14, 4 * i + branch] = sign
            p["Uh"][15, 4 * i + branch] = -sign
    p["Uh"][14, 28] = -2.0
    p["Uh"][14, 29] = 2.0
    p["Uh"][15, 28] = 2.0
    p["Uh"][15, 29] = -2.0

    p["Wh"][14, 14] = transport
    p["Wh"][14, 15] = -transport
    p["Wh"][15, 14] = -transport
    p["Wh"][15, 15] = transport
    p["Wc"][0, 32 + 14] = 1.0
    p["Wc"][0, 32 + 15] = 1.0

    # ---------------------------------------------------------------
    # Control-local click bonus.
    #
    # Detector magnitudes are exactly:
    #   large arrow:   t=1, |d|=offset
    #   compact arrow: t=3, |d|=2*offset
    #
    # Neighbouring transport-correction pixels have t <= 3/4.
    # Two hinges supply a valid-control bonus and cancel |d|.
    # ---------------------------------------------------------------
    valid_bonus = 1024.0
    slope_first = 8.0 * (valid_bonus + offset)
    slope_after = offset / 2.0

    for out, knot, coefficient in (
        (30, 7.0 / 8.0, slope_first),
        (31, 1.0, slope_after - slope_first),
    ):
        for i in range(7):
            w3(out, 14 + i, 1.0 / offset)
            w3(out, 21 + i, 1.0 / offset)
        p["b3"][out] = -knot
        p["Wc"][0, out] = coefficient

    # The click map makes the decision; memory is unused.
    p["bo"][6] = 1.0
    return p
