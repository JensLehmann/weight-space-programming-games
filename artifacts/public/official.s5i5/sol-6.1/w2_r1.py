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

    colours = (7, 8, 9, 10, 11, 12, 14)
    cardinal = ((-1, 0), (1, 0), (0, -1), (0, 1))
    eps = 1.0 / 256.0

    def a1(out, inp, dy, dx, value):
        p["W1"][out, inp, dy + 2, dx + 2] += value

    def a2(out, inp, dy, dx, value):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    def ui_pixel(out, dy, dx, value):
        a1(out, 2, dy, dx, value)
        a1(out, 4, dy, dx, value)

    # F1: four channels per colour.
    #   4c+0: right/down marker coordinate descriptor
    #   4c+1: decrement-control stem
    #   4c+2: left/up marker coordinate descriptor
    #   4c+3: increment-control stem
    q = 1024.0

    for ci, colour in enumerate(colours):
        base = 4 * ci

        for reverse, out in ((False, base), (True, base + 2)):
            inward = 1 if reverse else -1
            outward = -inward

            a1(out, 13, 0, 0, 10.0 * q)
            for dy, dx in cardinal:
                a1(out, colour, dy, dx, q)

            # A long rod supplies body two pixels inward.
            # A one-cell rod supplies its green pivot instead.
            for dy, dx in ((inward, 0), (0, inward)):
                a1(out, colour, 2 * dy, 2 * dx, q)
                a1(out, 3, dy, dx, 2.0 * q)

            blockers = (1, 3, 15) + tuple(
                c for c in colours if c != colour
            )
            for dy, dx in ((outward, 0), (0, outward)):
                for blocker in blockers:
                    a1(out, blocker, 2 * dy, 2 * dx, -4.0 * q)

            sign = -1.0 if reverse else 1.0
            a1(out, 16, 0, 0, sign)
            a1(out, 17, 0, 0, sign)
            constant = 6.0 if reverse else 4.0
            p["b1"][out] = -15.0 * q + constant

        # A genuine stem has two inward coloured diagonals.
        # This rejects the middle of each three-pixel glyph arm.
        # The definition handles both horizontal C glyphs and
        # vertical U glyphs.
        for increasing, out in (
            (False, base + 1),
            (True, base + 3),
        ):
            inward = -1 if increasing else 1
            outward = -inward

            a1(out, colour, 0, 0, 10.0)
            a1(out, colour, inward, inward, 10.0)
            a1(out, colour, -inward, inward, 10.0)
            a1(out, colour, inward, -inward, 10.0)

            ui_pixel(out, outward, outward, 10.0)
            a1(out, colour, outward, outward, -10.0)

            for dy, dx in cardinal:
                ui_pixel(out, dy, dx, -1.0)

            p["b1"][out] = -37.0

    # Uncovered target centre.
    for dy, dx in cardinal:
        a1(28, 13, dy, dx, 1.0)
        a1(29, 13, dy, dx, 10.0)
    a1(28, 13, 0, 0, -4.0)
    a1(29, 13, 0, 0, -40.0)
    p["b1"][28] = -3.0

    a1(29, 16, 0, 0, 1.0)
    a1(29, 17, 0, 0, 1.0)
    p["b1"][29] = -36.0

    # Rotation-control centre.
    for colour in colours:
        a1(30, colour, 0, 0, 10.0)
        for dy, dx in cardinal:
            a1(30, colour, dy, dx, 1.0)
    for dy in (-1, 1):
        for dx in (-1, 1):
            a1(30, 4, dy, dx, 1.0)
    p["b1"][30] = -17.0

    # F2: paired hinges. Their difference is one at a marker
    # and epsilon at the corresponding control.
    for ci in range(7):
        base = 4 * ci

        a2(base, base, 0, 0, 1.0)
        a2(base, base + 1, 0, 0, 3.0 + eps)
        p["b2"][base] = -3.0

        a2(base + 1, base, 0, 0, 1.0)
        p["b2"][base + 1] = -4.0

        a2(base + 2, base + 2, 0, 0, 1.0)
        a2(base + 2, base + 3, 0, 0, 3.0 + eps)
        p["b2"][base + 2] = -3.0

        a2(base + 3, base + 2, 0, 0, 1.0)
        p["b2"][base + 3] = -4.0

    a2(28, 28, 0, 0, 1.0)
    a2(29, 29, 0, 0, 1.0)

    for ci in range(7):
        a2(30, 4 * ci + 1, 0, 0, 1.0)
        a2(30, 4 * ci + 3, 0, 0, 1.0)
    a2(30, 30, 0, 0, 1.0)
    a2(31, 30, 0, 0, 1.0)

    def local(out, feature, value):
        p["W3"][out, feature] += value

    def local_presence(out, low, high, value):
        local(out, low, value)
        local(out, high, -value)

    def axial_union(out, feature, value):
        p["W3"][out, feature] -= value
        p["W3"][out, 32 + feature] += value
        p["W3"][out, 64 + feature] += value

    def target_coordinate(out, value):
        axial_union(out, 29, value)
        axial_union(out, 28, -4.0 * value)

    # F3: forward vote, reverse vote, and baseline per colour.
    # Subtracting the baseline recovers a signed resize vote.
    marker_gate = 64.0
    target_gate = 8.0
    offset = 2.0

    for ci in range(7):
        f = 4 * ci
        forward = 3 * ci
        reverse = forward + 1
        baseline = forward + 2

        local_presence(forward, f, f + 1, marker_gate)
        axial_union(forward, 28, target_gate)
        target_coordinate(forward, 1.0)
        local(forward, f + 1, -1.0)
        p["b3"][forward] = offset - marker_gate - target_gate

        local_presence(reverse, f + 2, f + 3, marker_gate + 2.0)
        local(reverse, f + 3, -1.0)
        axial_union(reverse, 28, target_gate)
        target_coordinate(reverse, -1.0)
        p["b3"][reverse] = offset - marker_gate - target_gate

        local_presence(baseline, f, f + 1, marker_gate)
        local_presence(baseline, f + 2, f + 3, marker_gate)
        axial_union(baseline, 28, target_gate)
        p["b3"][baseline] = offset - marker_gate - target_gate

    # Binary labels: codes 1..14 identify resize controls;
    # code 15 identifies rotation controls.
    for bit in range(4):
        out = 21 + bit

        for ci in range(7):
            f = 4 * ci
            decrement_code = 2 * ci + 1
            increment_code = 2 * ci + 2

            if (decrement_code >> bit) & 1:
                local_presence(out, f, f + 1, 1.0)
            if (increment_code >> bit) & 1:
                local_presence(out, f + 2, f + 3, 1.0)

        local(out, 30, 4.0)
        local(out, 31, eps)
        p["b3"][out] = -4.0

    local(25, 30, 1.0)
    local(26, 31, 1.0)

    # H: fourteen independently diffusing resize preferences.
    # A mismatching control supplies a large negative drive;
    # open space and matching controls supply exactly zero.
    inhibition = 16384.0

    for ci in range(7):
        forward = 3 * ci
        reverse = forward + 1
        baseline = forward + 2

        for increasing in (True, False):
            h = 2 * ci + (0 if increasing else 1)
            code = 2 * ci + (2 if increasing else 1)
            sign = 1.0 if increasing else -1.0
            population = sum(
                (code >> bit) & 1 for bit in range(4)
            )

            p["Wh"][h, h, :, :] = np.float32(1.0 / 9.0)
            p["Uh"][h, forward] = sign
            p["Uh"][h, reverse] = sign
            p["Uh"][h, baseline] = -sign

            p["Uh"][h, 25] = -inhibition * population
            for bit in range(4):
                coefficient = (
                    1.0 if ((code >> bit) & 1) else -1.0
                )
                p["Uh"][h, 21 + bit] = (
                    inhibition * coefficient / eps
                )

    # Click readout is restricted to recognized controls.
    p["bh"][14] = 1.0
    p["Uh"][14, 25] = -1.0
    p["Uh"][15, 26] = 1.0

    p["Wc"][0, 32:46] = 1.0
    p["Wc"][0, 46] = -10000.0
    p["Wc"][0, 47] = np.float32(1.0e-30)

    p["bo"][6] = 1.0
    return p
